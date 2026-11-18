# Mario long-run training launch marker

This marker launches the first 1,000-episode Double DQN training chunk on the real
Super Mario Bros emulator. It uses seed 42 and allows up to 20,000 environment
steps per episode. The workflow saves a full resumable checkpoint and disk-backed
replay memory, then runs a separate 20-episode greedy evaluation. It uploads the
checkpoint, replay memory, training CSV, and evaluation CSV as the
`mario-long-training-state` artifact.

Continue toward 40,000 episodes by starting **Actions → Mario Double DQN Long
Training** with `resume=true` and the previous workflow run ID. Preserve seed 42.
