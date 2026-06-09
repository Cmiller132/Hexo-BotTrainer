"""Dense CNN Model 1 tensor constants.

These values define the Python side of the model contract. The Rust encoder has
matching values in `rust/src/constants.rs`; any change to board size, channel
count, value bins, or plane order must update both files and the tests that
exercise the Python/Rust boundary.
"""

BOARD_SIZE = 41
BOARD_AREA = BOARD_SIZE * BOARD_SIZE
INPUT_CHANNELS = 13
VALUE_BINS = 65
DEFAULT_CHANNELS = 96
DEFAULT_BLOCKS = 6

PLANE_OWN_STONES = 0
PLANE_OPPONENT_STONES = 1
PLANE_EMPTY = 2
PLANE_LEGAL = 3
PLANE_SECOND_PLACEMENT = 4
PLANE_FIRST_STONE = 5
PLANE_PLAYER_COLOUR = 6
PLANE_OWN_RECENCY = 7
PLANE_OPPONENT_RECENCY = 8
PLANE_OPPONENT_HOT = 9
PLANE_OWN_HOT = 10
PLANE_CENTER_DISTANCE = 11
PLANE_OPPONENT_LAST_TURN = 12
