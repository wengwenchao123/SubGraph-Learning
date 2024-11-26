from math import log2



def calc_entropy(num, lst):

    return -sum([log2(n/num)*n/num if n!=0 else 0 for n in lst])



V0 = calc_entropy(15, [6, 9])

print(V0)



lst1 = [(6,[3,3]), (6,[3,3]), (3,[0,3])]

V1 = sum([calc_entropy(num, lst)*num/15 for num,lst in lst1])

print('根节点处，年龄的信息增益', V0-V1)



lst2 = [(5,[3,2]), (6,[2,4]), (4,[0,4])]

V2 = sum([calc_entropy(num, lst)*num/15 for num,lst in lst2])

print('根节点处，收入的信息增益', V0-V2)



lst3 = [(8,[4,4]), (7,[2,5])]

V3 = sum([calc_entropy(num, lst)*num/15 for num,lst in lst3])

print('根节点处，信用等级的信息增益', V0-V3)