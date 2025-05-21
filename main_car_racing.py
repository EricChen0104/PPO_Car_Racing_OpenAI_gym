import os
import gym
import numpy as np
import torch as T
import imageio
from ppo_torch import Agent
from utils import plot_learning_curve
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def save_gif(frames, filename, fps=30):
    """
    將畫面序列儲存為 GIF
    :param frames: 畫面列表（RGB 陣列）
    :param filename: GIF 檔案路徑
    :param fps: 每秒幀數
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with imageio.get_writer(filename, mode='I', fps=fps, format='GIF') as writer:
        for frame in frames:
            writer.append_data(frame)

def stack_frames(frame_buffer):
    """
    將緩衝區中的畫面堆疊為一個觀察
    :param frame_buffer: 包含最近四個畫面的列表
    :return: 形狀為 (12, 96, 96) 的堆疊觀察
    """
    return np.concatenate(frame_buffer, axis=0)

def test_agent(agent, env, n_test_episodes=10, render=True, save_gifs=True):
    """
    測試代理在環境中的表現，並可選擇儲存 GIF
    """
    test_scores = []
    agent.load_models()
    
    for i in range(n_test_episodes):
        observation, info = env.reset()
        observation = observation.astype(np.float32) / 255.0
        observation = np.transpose(observation, (2, 0, 1))  # (3, 96, 96)
        
        # 初始化畫面緩衝區，重複第一個畫面四次
        frame_buffer = [observation] * 4
        stacked_observation = stack_frames(frame_buffer)  # (12, 96, 96)
        
        done = False
        score = 0
        frames = []
        
        while not done:
            action, _, _ = agent.choose_action(stacked_observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            
            # 處理新畫面
            observation_ = observation_.astype(np.float32) / 255.0
            observation_ = np.transpose(observation_, (2, 0, 1))  # (3, 96, 96)
            frame_buffer.pop(0)  # 移除最舊的畫面
            frame_buffer.append(observation_)  # 添加新畫面
            stacked_observation = stack_frames(frame_buffer)  # 更新堆疊觀察
            
            if render or save_gifs:
                frame = env.render()
                if save_gifs:
                    frames.append(frame)
        
        if save_gifs:
            gif_filename = f'plots/test_episode_{i+1}.gif'
            save_gif(frames, gif_filename)
            print(f'儲存測試回合 {i+1} 的 GIF 至 {gif_filename}')
        
        test_scores.append(score)
        print(f'測試回合 {i+1}, 分數: {score:.1f}')
    
    avg_test_score = np.mean(test_scores)
    print(f'平均測試分數（{n_test_episodes} 回合）: {avg_test_score:.1f}')
    
    return test_scores

if __name__ == '__main__':
    # 初始化環境和代理
    env = gym.make('CarRacing-v2', domain_randomize=False, render_mode='rgb_array')
    N = 2048
    batch_size = 128
    n_epochs = 10
    alpha = 1e-4
    # 修改輸入維度為 (12, 96, 96) 以容納四個畫面
    agent = Agent(n_actions=3, 
                  batch_size=batch_size,
                  alpha=alpha, 
                  n_epochs=n_epochs,
                  input_dims=(12, 96, 96))  # 4 frames * 3 channels = 12
    n_games = 1500

    figure_file = 'plots/cartpole.png'

    best_score = -10000
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0

    prev_action = np.zeros(3)
    turning_streak = 0
    turning_direction = 0
    turning_threshold = 15

    max_steps = 1000 // 4
    '''
    # 訓練階段
    for i in range(n_games):
        observation, info = env.reset()
        observation = observation.astype(np.float32) / 255.0
        observation = np.transpose(observation, (2, 0, 1))  # (3, 96, 96)
        
        # 初始化畫面緩衝區
        frame_buffer = [observation] * 4
        stacked_observation = stack_frames(frame_buffer)  # (12, 96, 96)
        
        done = False
        score = 0
        turning_streak = 0
        turning_direction = 0
        frames = []
        
        brakes_list = [0] * 50
        brakes_cnt = 0
        
        while not done:
            action, prob, val = agent.choose_action(stacked_observation)
            # action = np.array([action[0], max(action[1], 0), max(action[2], 0)])
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            n_steps += 1
            
            # 處理新畫面
            observation_ = observation_.astype(np.float32) / 255.0
            observation_ = np.transpose(observation_, (2, 0, 1))  # (3, 96, 96)
            frame_buffer.pop(0)  # 移除最舊的畫面
            frame_buffer.append(observation_)  # 添加新畫面
            next_stacked_observation = stack_frames(frame_buffer)  # 更新堆疊觀察
            
            # if action[1] > 0.5: reward += 0.05 * action[1]
            score += reward
            
            frame = env.render()
            frames.append(frame)
            
            # 儲存記憶，注意使用堆疊觀察
            agent.remember(stacked_observation, action, prob, val, reward, done)
            
            stacked_observation = next_stacked_observation

        agent.learn()
        
        if (i + 1) % 100 == 0:
            gif_filename = f'plots/train_episode_{i+1}.gif'
            save_gif(frames, gif_filename)
            print(f'儲存訓練回合 {i+1} 的 GIF 至 {gif_filename}')
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        if score > best_score:
            best_score = score
            gif_filename = f'plots/best_play{i+1}.gif'
            save_gif(frames, gif_filename)
            agent.save_models()
        
        print('episode', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    '''
    # 測試階段
    print("\n開始測試訓練好的代理...")
    test_scores = test_agent(agent, env, n_test_episodes=10, render=True, save_gifs=True)

    env.close()