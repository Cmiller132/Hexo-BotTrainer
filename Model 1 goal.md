The goal is to create a simple self play model and create a good process for training epochs and debugging / working on the model.

Specs are in Model 1 goal.md


CRITICAL DETAIL:
Implement D6 augmentation on the sample data. At training time apply a random augmentation to each sample.
Samples should be compact and compressed(compress training samples in ram until they are needed)
Training selects X samples from the buffer and then expands those into full uncompressed dense targets at training time.
When building these targets apply random D6 AUG. Make sure it is well tested and implemented in a robust way. we need to be sure that the model is not being subtly poisoned.

Below are a list of stages and goals


GOAL #1

Get a version of this model that works!!!
It can be a small model with something like 6 blocks and 96 wide channels to start out with.
It needs to have each individual part work and have working inference, training, self play, and sample creation.
Do your best to verify that the model works and is a good starting point. Make sure all specs are matched before moving on to goal 2.
Also for self play and training do it sequentially. Do x games and then train on the samples. Test a few values but a good baseline would be 4096 self play samples per epoch and then train on a random selection of 4096 samples from the buffer. Buffer should store at least 200k samples, ideally more. Select 4096 randomly with recency decay. Then train the model on those and move on to the next epoch. 

Make sure that each epoch saves a checkpoint of the model so that crashes / code changes dont require a full restart.

Also each epoch do 64 eval games against sealbot best 50ms. This is simply for record keeping but it should show if the model starts to improve eventually.

Goal #2

Make it fast. Now that we have a working model, optimize training, inference, and memory usage. Some tips would be batch MCTS leaves for inference, use amp, etc. Also have training batch size be optimized. It should be possible to have a calibration step where the model automatically picks the best settings for MCTS batching, Inference, Training batch size, etc. This is critical and this step will take a long time. Make sure the model is able to hit at least 128 pos/s in self play. Keep iterating and benchmarking until the model is optimized fully and able to run at peak performance on this machine. It should also automatically tune the settings for different sized models, we want to automate performance tweaking to get the best option. Also make sure training is quick. Prefer to offload memory pressure to the cpu as my 7950x has 16 cores. Multithread where possible even if difficult. Make sure GPU and CPU usage is maxed out and time is not being wasted.

Get at least a 64x4 model running at 128 pos/s
Dont do tweaks to optimize, find the root cause and attempt to improve the performance at a root level even if that means rewriting expensive code in rust. USE KATAGO AS A GREAT BASELINE, any idea you have Katago likely has a similar or better solution. Also use version control to test and iterate. If something doesn't work remove it, dont keep bloating the codebase 

It needs to get 128 pos/s with 128 MCTS sims
Goal #3
Make it easy to debug. Expand the frontend dashboard and logging to keep game history records, keep policy targets and any other useful debug info. Make it easy to access and stored in a natural easy to work with way.

Goal #3a
Regularly check in on the dashboard and make it work on mobile, desktop, lan etc. Also add features regularly and make it easier to browse. also regularly test it and use it to verify info. Use the dashboard yourself so that you know that it is usable. 

Goal #4
Make it good. Tune and train the model for many epochs(20-30) until it can hold its own against sealbot 50ms. It needs to be able to reliably beat sealbot. You can check for progress by looking at game length in the evaluation. If the model is improving it will increase survival time but if it stays stuck at 20-30 moves it does not have a strong strategy at all.

Intervene and make sure the games look coherent and not random. If after epoch 6 or so games look random(view them via a png) then the model is not learning. Make sure that the model is not subtly broken and keeps improving.

Goal #5
Take your model that beats sealbot and do any finishing touches, clean up all unused code and experiments you did. make the codebase clean and cohesive. Do not leave legacy support or complicated wrappers. Rewrite messy code. Make the codebase clean, maintainable and designed around a single clear production path to train and test models.

Goal #6
Take the model that beats sealbot and keep iterating. Make it slightly larger, improve weak areas. document it in the dashboard. Keep working at it until the model is as good as possible. We need at least 72 hours of training.
