
import yaml
import cv2
import pyautogui
import time
from yaml import *
from action import *
from DQN import *
from find_blood import *
from grabscreen import *
from log import *


if __name__ == "__main__":
    agent = DQN(HEIGHT, WIDTH, action_size, DQN_model_path, DQN_log_path)  # 初始化DQN代理
    print("start")  # 打印开始信息
    for episode in range(yaml.EPISODES):  # 循环训练轮数
        screen_gray = cv2.cvtColor(grab_screen(x3, y3, width3, height3), cv2.COLOR_BGR2GRAY)  # 获取屏幕截图并转换为灰度图
        state = cv2.resize(screen_gray, (WIDTH, HEIGHT))  # 调整状态图像尺寸
        # 截取第一个区域
        region_1 = pyautogui.screenshot(region=(x, y, width, height))
        region_1_cv = pil_to_cv(region_1)
        boss_blood= recognize_digits(region_1_cv)   
        # 截取第二个区域
        region_2 = pyautogui.screenshot(region=(x2, y2, width2, height2))
        region_2_cv = pil_to_cv(region_2)
        self_blood = recognize_digits(region_2_cv)
      
        target_step = 0  # 重置目标步骤计数
        done = 0  # 设置完成标志
        total_reward = 0  # 初始化累计奖励
        stop = 0  # 设置停止标志
        emergence_break = 0  # 设置紧急停止次数
        last_time = time.time()  # 获取当前时间
        # state1 = torch.tensor(state, dtype=torch.float).to(mps)  # 将状态转换为张量并移动到指定设备
        
        while True:  # 主循环
            state = state.reshape(-1, 1, HEIGHT, WIDTH)[0]  # 调整状态张量形状
            survival_time=time.time() - last_time
            print(f'loop took {survival_time} seconds')  # 打印循环耗时
            target_step += 1  # 增加目标步骤计数
            action = agent.choose_action(state)  # 选择动作
            take_action(action)  # 执行动作
            #下一状态动作
            screen_gray = cv2.cvtColor(grab_screen(x3, y3, width3, height3), cv2.COLOR_RGB2GRAY)  # 获取新屏幕截图并转换为灰度图
                
            region_1 = pyautogui.screenshot(region=(x, y, width, height))
            region_1_cv = pil_to_cv(region_1)
            next_boss_blood= recognize_digits(region_1_cv)   
            # 截取第二个区域
            region_2 = pyautogui.screenshot(region=(x2, y2, width2, height2))
            region_2_cv = pil_to_cv(region_2)
            next_self_blood = recognize_digits(region_2_cv)
      
      
            # my_blood_window_gray = grab_screen(x, y, width, height)  # 获取新的玩家血条区域
            
            if next_boss_blood=='' and next_self_blood=='':  # 检查血条是否有效
                cv2.waitKey(8000)
                pyautogui.press('enter')
                cv2.waitKey(2000)
                pyautogui.press('up')
                cv2.waitKey(2000)
                pyautogui.press('enter')
                cv2.waitKey(5000)  # 等待一段时间
                screen_gray = cv2.cvtColor(grab_screen(x3, y3, width3, height3), cv2.COLOR_RGB2GRAY)
                region_1 = pyautogui.screenshot(region=(x, y, width, height))
                region_1_cv = pil_to_cv(region_1)
                next_boss_blood= recognize_digits(region_1_cv)   
                # 截取第二个区域
                region_2 = pyautogui.screenshot(region=(x2, y2, width2, height2))
                region_2_cv = pil_to_cv(region_2)
                next_self_blood = recognize_digits(region_2_cv)
        
            next_state = cv2.resize(screen_gray, (WIDTH, HEIGHT))  # 调整下一状态图像尺寸
            next_state = np.array(next_state).reshape(-1, 1, HEIGHT, WIDTH)[0]  # 调整下一状态张量形状
            print("next_boss_blood: ", next_boss_blood)  # 打印下一Boss血量
            print("next_self_blood: ", next_self_blood)  # 打印下一玩家血量
            reward, done, stop, emergence_break = action_judge(boss_blood, next_boss_blood,
                                                               self_blood, next_self_blood,
                                                               stop, emergence_break,survival_time)  # 判断动作效果
            if emergence_break == 100:  # 如果达到紧急停止条件
                print("emergence_break")  # 打印紧急停止信息
                agent.save_model()  # 保存模型
                paused = True  # 设置暂停标志
          
            agent.store_data(state, action, reward, next_state, done)  # 存储数据
            if len(agent.replay_buffer) > big_BATCH_SIZE:  # 如果经验池足够大
                num_step += 1  # 增加步数计数
                agent.train()  # 训练网络
            if target_step % UPDATE_STEP == 0:  # 如果达到更新步数R
                
                agent.update_target()  # 更新目标网络
            
            
            state = next_state 
            
            # 更新状态
            self_blood = next_self_blood  # 更新玩家血量
            boss_blood = next_boss_blood  # 更新Boss血量
            total_reward += reward  # 累加奖励
            if done == 1:  # 如果完成
                log_info(f'episode: {episode}, Evaluation Average Reward: {total_reward / target_step},{total_reward}')
                break  # 结束循环
            if episode % 10 == 0:  # 如果达到保存条件
                agent.save_model()  # 保存模型
        print('episode: ', episode, 'Evaluation Average Reward:', total_reward / target_step,total_reward)  # 打印评价信息
 