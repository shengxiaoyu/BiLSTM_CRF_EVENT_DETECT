#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'

from gurobipy import *

def optimize(length,num_labels,trans,logits,id2tag,trigger_ids,trigger_args_dict,i_ids):
    '''

    :param length: 句长
    :param num_labels: 标签数量
    :param trans: crf转移矩阵
    :param logits: blstm经过全连接层生成的置信分数
    :param id2tag: id转标签
    :param trigger_ids: B_触发词类标签id
    :param trigger_args_dict:  每类B_触发词标签对应的B_参数标签id
    :return:
    '''
    try:
        m = Model('ilp-solver')

        '''You can use the PoolSearchMode parameter to control the approach used to find solutions. In its default setting (0),
         the MIP search simply aims to find one optimal solution. Setting the parameter to 1 causes the MIP search to expend 
         additional effort to find more solutions, but in a non-systematic way. You will get more solutions, but not necessarily 
         the best solutions. Setting the parameter to 2 causes the MIP to do a systematic search for the n best solutions. For 
         both non-default settings, the PoolSolutions parameter sets the target for the number of solutions to find.'''
        m.Params.poolSearchMode = 2
        # m.Params.PoolSolutions = 200
        '''If you are only interested in solutions that are within a certain gap of the best solution found, you can set the PoolGap 
        parameter. Solutions that are not within the specified gap are discarded.PoolGap是根据beat solution的比例'''
        m.Params.PoolGap = 0.15

        distributions = m.addVars(length-1,num_labels,num_labels,vtype=GRB.BINARY,name="distributions")

        obj = LinExpr()
        for i in range(length-1):
            for l in range(num_labels):
                for h in  range(num_labels):
                    obj += distributions[i,l,h]*(logits[i][l]+trans[l][h])
        m.setObjective(obj,GRB.MAXIMIZE)

        #C1:each word should be and only be annotated with one label
        m.addConstrs((distributions.sum(i,'*','*')==1 for i in range(length-1)),name='con1')

        #C2:一个v成立，则一定有下一个v成立
        m.addConstrs(distributions.sum(i,'*',h)==distributions.sum(i+1,h,'*') for i in range(length-2) for h in range(num_labels))

        #C3:一个标记为I_label,则上一个要么事I_label,要么是B_label
        m.addConstrs( distributions.sum(i,h,'*')==distributions[i-1,h-1,h]+distributions[i-1,h,h] for i in range(1,length-1) for l in i_ids)

        #C4：必须出现触发词才能出现事件的参数,即触发词出现的次数大于参数出现的次数
        m.addConstrs(distributions.sum('*',trigger_id,'*')>=distributions.sum('*',arg_id,'*') for trigger_id in trigger_ids for arg_id in trigger_args_dict[trigger_id] )

        if(m.isMIP==1):
            print("是MIP")
        else:
            print("不是MIP")

        #开始优化
        m.optimize()

        if(m.status == GRB.Status.OPTIMAL):
            nSolutions = m.solCount
            print('总共标记方案：'+str(nSolutions))

            objVals = []
            ids_list = []

            objVal = m.objVal
            print('最优解：'+str(objVal))
            solution = m.getAttr('x',distributions)
            ids = get_result(length,num_labels,solution)

            objVals.append(objVal)
            ids_list.append(ids)

            for i in range(nSolutions):
                m.Params.solutionNumber = i

                #得分
                objVal = m.PoolObjVal
                #方案
                solution = m.getAttr('xn',distributions)
                ids =  get_result(length,num_labels,solution)
                print('Solution %d has objective %g ' % (i,objVal))
                print(' '.join([str(id) for id in ids]))
                objVals.append(objVal)
                ids_list.append(ids)
            return objVals,ids_list
        else:
            return None,None

    except GurobiError as e:
        print('Error code'+str(e.errno)+':'+str(e))
    except AttributeError as e:
        print('some error:'+str(e))

def get_result(length,num_labelds,x):
    result = []
    for i in range(length-1):
        for l in range(num_labelds):
            for h in range(num_labelds):
                if(x[i,l,h]==1):
                    if(i==0):
                        result.append(l)
                    result.append(h)
    return result

if __name__ == '__main__':
    pass