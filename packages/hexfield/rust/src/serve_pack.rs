//! Rust parallel serve-pack with zero-copy buffers (the dense_cnn PlaneBuffer
//! move applied to the hexfield serve path).
//!
//! Eliminates the Python per-group PADDING loop on the serve GIL thread (the
//! remaining inter-forward host gap; GPU was 33-46% idle post-FlexAttention).
//! `build_serve_groups` takes the CSR-flat request (already on the wire as f16
//! feats / i16 coords / u16 nbr) plus the per-row offsets, runs the IDENTICAL
//! `plan_groups` boundary planner, and assembles the padded per-group buffers
//! in PARALLEL (rayon), exposing them as read-only zero-copy `#[pyclass]`
//! buffers consumed Python-side via `torch.frombuffer(buf, ...).to(device)` —
//! NO Python pack loop, NO astype.
//!
//! BIT-EXACT to the Python `_forward_group` pad: this is PURE DATA MOVEMENT
//! (same f16 feats, same int values, same padding) so the forward inputs are
//! byte-identical to the Python pack and the serve outputs are bit-identical.
//! The exact pad rules reproduced:
//!   - feats pad rows  = 0  (`f16::ZERO`)
//!   - nbr fill        = `pad_to`; `NBR_SENTINEL (0xFFFF)` -> `pad_to`
//!   - mask            = `1` only at real nodes, else `0`
//!   - coords          = 0 at pad rows (`np.zeros` in Python; never read but
//!                       zero-filled to meet the byte-identical bar)
//! The model gathers need int64 nbr/coords on-device; Rust emits int32 (sound:
//! payload.rs hard-asserts `num_nodes <= NBR_SENTINEL`, so every index fits i32
//! with vast headroom) and Python casts to int64 on the GPU, value-preserving.

use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

use half::f16;

use crate::constants::NUM_FEATURES;

const NBR_SENTINEL: u16 = 0xFFFF;
// Must mirror inference.py EXACTLY (the §D boundary-equivalence test gates this).
const QUANT_NODES: usize = 64;
// NUM_TOKENS in the (pad_to + NUM_TOKENS)^2 pair-ceiling test (model has 8 tokens).
const NUM_TOKENS: usize = 8;
const PAIR_CEILING: f64 = 3.8e7;
// f64 literal — must be the literal 0.18_f64 so int(0.18*pad_to) truncates
// identically to CPython's int(WASTE_FRACTION * pad_to).
const WASTE_FRACTION: f64 = 0.18;

/// `inference.py::_ceil_quant`: `max(64, -(-n//64)*64)`. For n>=1 this is the
/// 64-quantized ceiling; for n==0 it is 64 (matches CPython's `-(-0//64)*64=0`
/// then `max(64,0)=64`).
fn ceil_quant(n: usize) -> usize {
    let q = n.div_ceil(QUANT_NODES) * QUANT_NODES;
    QUANT_NODES.max(q)
}

/// Port of `inference.py::plan_groups`. Rows arrive size-DESCENDING. Returns
/// (start, end, pad_to) groups in ascending row order. BYTE-FOR-BYTE boundary
/// parity with the Python planner is a HARD GATE (debug_plan_groups test).
pub fn plan_groups(sizes: &[usize]) -> Vec<(usize, usize, usize)> {
    let n = sizes.len();
    let mut groups: Vec<(usize, usize, usize)> = Vec::new();
    let mut start = 0usize;
    while start < n {
        let pad_to = ceil_quant(sizes[start]);
        // floor = pad_to - max(QUANT_NODES, int(WASTE_FRACTION * pad_to))
        // int(...) truncates toward zero; pad_to >= 0 so `as usize` truncates
        // the (non-negative) f64 product identically to CPython int().
        let waste = (WASTE_FRACTION * pad_to as f64) as usize;
        let floor = pad_to - QUANT_NODES.max(waste);
        let mut end = start + 1;
        while end < n {
            // (end - start + 1) is the candidate group size INCLUDING sizes[end].
            let g = (end - start + 1) as f64;
            let pair = g * ((pad_to + NUM_TOKENS) * (pad_to + NUM_TOKENS)) as f64;
            if pair > PAIR_CEILING {
                break;
            }
            if sizes[end] < floor {
                break;
            }
            end += 1;
        }
        groups.push((start, end, pad_to));
        start = end;
    }
    groups
}

// --- Zero-copy buffers (exact PlaneBuffer ABI) --------------------------------

macro_rules! plane_buffer {
    ($name:ident, $ty:ty) => {
        #[pyclass]
        pub struct $name {
            data: Vec<$ty>,
        }

        #[pymethods]
        impl $name {
            fn __len__(&self) -> usize {
                self.data.len() * std::mem::size_of::<$ty>()
            }

            /// SAFETY: populate the CPython-supplied `Py_buffer` with a read-only
            /// 1-D byte view over `data`, keeping `slf` alive via `view.obj`.
            unsafe fn __getbuffer__(
                slf: Bound<'_, Self>,
                view: *mut ffi::Py_buffer,
                flags: c_int,
            ) -> PyResult<()> {
                if view.is_null() {
                    return Err(PyBufferError::new_err("buffer view is null"));
                }
                if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
                    (*view).obj = ptr::null_mut();
                    return Err(PyBufferError::new_err("buffer is read-only"));
                }
                let guard = slf.borrow();
                let data = &guard.data;
                (*view).buf = data.as_ptr() as *mut c_void;
                (*view).len =
                    (data.len() * std::mem::size_of::<$ty>()) as ffi::Py_ssize_t;
                (*view).readonly = 1;
                (*view).itemsize = 1;
                (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
                    b"B\0".as_ptr() as *mut c_char
                } else {
                    ptr::null_mut()
                };
                (*view).ndim = 1;
                (*view).shape = ptr::null_mut();
                (*view).strides = ptr::null_mut();
                (*view).suboffsets = ptr::null_mut();
                (*view).internal = ptr::null_mut();
                (*view).obj = slf.clone().into_any().into_ptr();
                Ok(())
            }

            unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
        }
    };
}

plane_buffer!(F16Buf, f16); // feats:  g*pad_to*NUM_FEATURES
plane_buffer!(I32Buf, i32); // nbr: g*pad_to*6 ; coords: g*pad_to*2
plane_buffer!(U8Buf, u8); //  mask:   g*pad_to (0/1)

/// One assembled group's four buffers + shape, ready to hand to Python.
struct GroupBufs {
    start: usize,
    end: usize,
    pad_to: usize,
    g: usize,
    feats: Vec<f16>,
    nbr: Vec<i32>,
    mask: Vec<u8>,
    coords: Vec<i32>,
}

/// Assemble all groups' padded buffers in PARALLEL (one worker per group, good
/// locality), reproducing the Python `_forward_group` pad EXACTLY. GIL-free.
///
/// `feats` is the CSR-flat f16 node features (total_nodes * NUM_FEATURES),
/// `qr` the i16 coords (total_nodes * 2), `nbr` the u16 row-local neighbours
/// (total_nodes * 6, sentinel 0xFFFF), `offsets` the per-row node offsets
/// (len b+1), `sizes` the per-row node counts (len b). All in sorted (DESC)
/// row order, matching the wire.
fn assemble_groups(
    groups: &[(usize, usize, usize)],
    feats: &[f16],
    qr: &[i16],
    nbr: &[u16],
    offsets: &[usize],
    sizes: &[usize],
) -> Vec<GroupBufs> {
    groups
        .par_iter()
        .map(|&(start, end, pad_to)| {
            let g = end - start;
            let pad_i32 = pad_to as i32;

            let mut f = vec![f16::ZERO; g * pad_to * NUM_FEATURES];
            let mut nb = vec![pad_i32; g * pad_to * 6];
            let mut m = vec![0u8; g * pad_to];
            let mut c = vec![0i32; g * pad_to * 2];

            for k in 0..g {
                let row = start + k;
                let n = sizes[row];
                let o = offsets[row];

                // feats: [k, :n, :] = feats[o:o+n]; rest already f16::ZERO.
                let src = &feats[o * NUM_FEATURES..(o + n) * NUM_FEATURES];
                let dst = &mut f[k * pad_to * NUM_FEATURES..k * pad_to * NUM_FEATURES + n * NUM_FEATURES];
                dst.copy_from_slice(src);

                // nbr: [k, :n, :] = where(row==SENTINEL, pad_to, row); rest = pad_to.
                let nsrc = &nbr[o * 6..(o + n) * 6];
                let nbase = k * pad_to * 6;
                for (i, &j) in nsrc.iter().enumerate() {
                    nb[nbase + i] = if j == NBR_SENTINEL { pad_i32 } else { j as i32 };
                }

                // mask: [k, :n] = 1; rest already 0.
                for i in 0..n {
                    m[k * pad_to + i] = 1;
                }

                // coords: [k, :n, :] = qr[o:o+n]; rest already 0.
                let qsrc = &qr[o * 2..(o + n) * 2];
                let cbase = k * pad_to * 2;
                for (i, &v) in qsrc.iter().enumerate() {
                    c[cbase + i] = v as i32;
                }
            }

            GroupBufs {
                start,
                end,
                pad_to,
                g,
                feats: f,
                nbr: nb,
                mask: m,
                coords: c,
            }
        })
        .collect()
}

/// Flag-gated alternate arm (HEXFIELD_RUST_PACK, off by default). Parse the
/// CSR-flat request, plan the IDENTICAL groups, assemble the padded per-group
/// buffers in parallel (GIL released), then build the Python group dict list.
///
/// `feats_bytes`: f16 LE, total_nodes*NUM_FEATURES. `qr_bytes`: i16 LE,
/// total_nodes*2. `nbr_bytes`: u16 LE, total_nodes*6. `offsets`: i64 row
/// offsets (len b+1, offsets[-1]==total_nodes). Returns a list of dicts:
/// {start,end,pad_to,g, feats:F16Buf, nbr:I32Buf, mask:U8Buf, coords:I32Buf}.
#[pyfunction]
pub fn build_serve_groups<'py>(
    py: Python<'py>,
    feats_bytes: &[u8],
    qr_bytes: &[u8],
    nbr_bytes: &[u8],
    offsets: Vec<i64>,
) -> PyResult<Bound<'py, PyList>> {
    if offsets.is_empty() {
        return Err(PyValueError::new_err("offsets must be non-empty (b+1)"));
    }
    let b = offsets.len() - 1;
    let total_nodes = *offsets.last().unwrap() as usize;

    // Reinterpret the wire bytes as typed slices. These are LE/native on the
    // build+serve host (same machine) so a bytewise reinterpret is exact.
    if feats_bytes.len() != total_nodes * NUM_FEATURES * std::mem::size_of::<f16>() {
        return Err(PyValueError::new_err("feats byte count mismatch"));
    }
    if qr_bytes.len() != total_nodes * 2 * std::mem::size_of::<i16>() {
        return Err(PyValueError::new_err("qr byte count mismatch"));
    }
    if nbr_bytes.len() != total_nodes * 6 * std::mem::size_of::<u16>() {
        return Err(PyValueError::new_err("nbr byte count mismatch"));
    }
    // SAFETY: lengths checked above; f16/i16/u16 are plain POD with no invalid
    // bit patterns; the source byte buffers (PyBytes/numpy) are alive for the
    // call and properly aligned (PyBytes is malloc-aligned; element align <= 2
    // and these are contiguous arrays). Build owned Vecs (cheap parse copies).
    let feats: Vec<f16> = unsafe {
        std::slice::from_raw_parts(feats_bytes.as_ptr() as *const f16, total_nodes * NUM_FEATURES)
    }
    .to_vec();
    let qr: Vec<i16> = unsafe {
        std::slice::from_raw_parts(qr_bytes.as_ptr() as *const i16, total_nodes * 2)
    }
    .to_vec();
    let nbr: Vec<u16> = unsafe {
        std::slice::from_raw_parts(nbr_bytes.as_ptr() as *const u16, total_nodes * 6)
    }
    .to_vec();

    let offsets_us: Vec<usize> = offsets.iter().map(|&o| o as usize).collect();
    let sizes: Vec<usize> = (0..b).map(|i| offsets_us[i + 1] - offsets_us[i]).collect();

    let groups = plan_groups(&sizes);

    // Parallel pad with the GIL released (the host-side win).
    let assembled =
        py.detach(|| assemble_groups(&groups, &feats, &qr, &nbr, &offsets_us, &sizes));

    let out = PyList::empty(py);
    for gb in assembled {
        let d = PyDict::new(py);
        d.set_item("start", gb.start)?;
        d.set_item("end", gb.end)?;
        d.set_item("pad_to", gb.pad_to)?;
        d.set_item("g", gb.g)?;
        d.set_item("feats", Py::new(py, F16Buf { data: gb.feats })?)?;
        d.set_item("nbr", Py::new(py, I32Buf { data: gb.nbr })?)?;
        d.set_item("mask", Py::new(py, U8Buf { data: gb.mask })?)?;
        d.set_item("coords", Py::new(py, I32Buf { data: gb.coords })?)?;
        out.append(d)?;
    }
    Ok(out)
}

/// §D HARD-GATE surface: expose the SAME `plan_groups` used by the hot path so
/// the parity test can assert `debug_plan_groups(sizes) == inference.plan_groups`
/// element-for-element (start, end, pad_to).
#[pyfunction]
pub fn debug_plan_groups(sizes: Vec<usize>) -> Vec<(usize, usize, usize)> {
    plan_groups(&sizes)
}
