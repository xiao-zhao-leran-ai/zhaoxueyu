import directkeys

# 执行动作
def take_action(action):
    print("action: ", action)  # 打印执行的动作
    if action == 0:  # 不做任何操作
        pass
    elif action == 1:  
        directkeys.A()
    elif action == 2:  
        directkeys.S()
    elif action == 3:  
        directkeys.D()
    elif action == 4:  
        directkeys.W()
    elif action == 5:  
        directkeys.R()
    elif action == 6:  
        directkeys.left_click()
    elif action == 7: 
        directkeys.right_click()
    elif action == 8:  
        directkeys.press_space()
    elif action == 9:  
        directkeys.release_space()
    elif action == 10:  
        directkeys.B()
    elif action == 11:  
        directkeys.left_click()

# 判断动作的效果
def action_judge(boss_blood, next_boss_blood, self_blood, next_self_blood, stop, emergence_break,survival_time):
    # 获取动作奖励
    # emergence_break用于紧急停止训练以防止错误训练数据干扰神经网络
    if next_boss_blood=='' and next_self_blood=='':  # 玩家死亡
        if emergence_break < 2:
            reward = -2000  # 设置奖励
            done = 1  # 设置完成标志
            stop = 0  # 设置停止标志
            emergence_break += 1  # 增加紧急停止次数
            return reward, done, stop, emergence_break
        else:
            reward = -2000  # 设置奖励
            done = 1  # 设置完成标志
            stop = 0  # 设置停止标志
            emergence_break = 100  # 设置紧急停止次数上限
            return reward, done, stop, emergence_break
    else:
        self_blood_reward = 0  # 初始化玩家血量奖励
        survival_reward=0
        boss_blood_reward = 0  # 初始化Boss血量奖励
        if next_self_blood!=self_blood:  # 玩家血量减少过多
            if stop == 0:
                self_blood_reward =-6# 设置奖励
                stop = 1  # 设置停止标志
        if next_boss_blood!=boss_blood:  # Boss血量减少
            boss_blood_reward = 4  # 设置奖励
        if  next_self_blood==self_blood and next_boss_blood==boss_blood:
             survival_reward=0.1
        if  next_self_blood==self_blood and next_boss_blood!=boss_blood:
             boss_blood_reward=10     
        reward = self_blood_reward + boss_blood_reward+survival_reward # 计算总奖励
        done = 0  # 设置完成标志
        emergence_break = 0  # 重置紧急停止次数
        return reward, done, stop, emergence_break
