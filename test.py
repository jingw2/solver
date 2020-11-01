class Solution:
    """
    @param cards: A list of cards.
    @return: A list of feasible solution.
    """
    def getTheNumber(self, cards):
        # write your code here
        
        if cards is None or len(cards) == 0:
            return [0]
        
        from collections import Counter 
        counter = Counter(cards)
        
        ans = []
        for num in range(1, 10):
            if counter[num] == 4:
                continue 
            newCards = cards + [num]
            counter[num] = counter.get(num, 0) + 1
            if self.isValid(counter):
                ans.append(num)
            counter[num] -= 1 
        return ans 
    
    def isValid(self, counter):
        tmp = {i: 0 for i in range(1, 10)}
        # 先选出雀头, sparrow，剩下的要么3个，要么连着
        for i in range(1, 10):
            # 雀头至少两个
            if counter[i] < 2:
                continue 
            # copy 
            for num in counter:
                tmp[num] = counter[num]
            # 雀头去掉
            tmp[num] -= 2 
            valid = True 
            for j in range(1, 10):
                # 3个一样的，可以直接拼掉，如果4个的，肯定会拆
                if not valid:
                    break
                if tmp[j] >= 3:
                    tmp[j] -= 3
                # tmp[j] 还有的话就是组顺子
                while tmp[j] and valid:
                    if j > 7: # 大于7往后组顺子没戏
                        valid = False 
                        break 
                    if tmp[j+1] and tmp[j+2]:
                        tmp[j+1] -= 1 
                        tmp[j+2] -= 1 
                    else:
                        valid = False 
            it valid:
                return True  
        return False 
                