import time
import numpy as np
import torch
import gym
import model
import memory
import helpers
import conf

# Tensorboard logging
import tensorboard_logger as tboard
timestamp = str(round(time.time()))[-6:]
tb_dir = conf.DIR + "/runs/es_"+timestamp
tboard.configure(tb_dir, flush_secs=5)

# gym
env = gym.make("BreakoutDeterministic-v4")
conf.OUTPUT_NUM = env.action_space.n

# models and learning
encoder = model.Encoder()

inverse_model = model.InverseModel(encoder=encoder)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(inverse_model.parameters(), lr=0.1)#torch.optim.Adam(inverse_model.parameters(), lr=0.001)

# buffer
memory = memory.MemoryBuffer(size=20000, batch_size=8, replace=True, bias_prob=0.3)


curr_state = None
episode_i = 0

while True:
    env.reset()
    episode_i += 1

    loss_per_episode = []
    accuracy_per_episode = []
    for t in range(conf.EPISODE_LENGTH):
        action = env.action_space.sample()

        frames = []
        reward_for_step = 0
        # getting frames
        for ft in range(conf.FRAMES_PER_STATE):
            env.render()
            frame, reward, done, _ = env.step(action)
            reward_for_step += reward
            frames.append(frame)


        prev_state = curr_state
        curr_state = helpers.combine(frames)

        if prev_state is not None:
            # adding to memory
            reward_total = reward_for_step/conf.FRAMES_PER_STATE
            memory.add(prev_state,action,reward_total,curr_state)

        # LEARN #
        if len(memory.buffer) >= memory.size and t % 10 == 0:
            optimizer.zero_grad()
            batch = memory.get_batch()
            learn_prev_state = torch.tensor(batch["prev_state"]).to(torch.float32)*(1.0/255.0)
            learn_curr_state = torch.tensor(batch["curr_state"]).to(torch.float32)*(1.0/255.0)

            pred = inverse_model(learn_prev_state, learn_curr_state)
            target = torch.tensor(batch["prev_action"]).to(torch.long)

            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                loss_per_episode.append(float(loss))
                pred_actions = torch.argmax(pred, dim=1)
                match = pred_actions==target
                acc = float(torch.sum(match))/match.shape[0]

                accuracy_per_episode.append(acc)

        # END Episode #
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

    # Save and log every episdoe
    if episode_i % 20 == 0:
        torch.save(inverse_model.state_dict(), conf.DIR+"/models/EC_"+str(episode_i))

    ## loggging
    if len(loss_per_episode) > 0:
        mean_loss = sum(loss_per_episode)/len(loss_per_episode)
    else:
        mean_loss = 0

    tboard.log_value("Loss", mean_loss, episode_i)

    if len(accuracy_per_episode) > 0:
        mean_acc = sum(accuracy_per_episode)/len(accuracy_per_episode)
    else:
        mean_acc = 0

    tboard.log_value("Accuracy", mean_acc, episode_i)
