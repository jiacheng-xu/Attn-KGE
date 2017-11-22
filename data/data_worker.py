dict={}
dict_all = {}
list = []
with open('rm_overlap_des-jan25.txt','r') as f:
    lines = f.readlines()
    for i in xrange(len(lines)):
        try:
            mid = lines[i].split('\t')[0]
            if dict.has_key(mid):
                # print 'Overlap!'
                pass
            else:
                dict[mid] = i
                list.append(i)
        except:
            print i
            print 'Format error'

    for x in dict:
        if len(x)>10:
            print x

    # with open('rm_overlap_des-jan25.txt','a') as wrt:
    #     for j in list:
    #         wrt.write(lines[j])
    count = 0
    with open('/Users/jcxu/Desktop/Relation_Extraction/data/FB15k/entity2id.txt','r') as rd:
        ls = rd.readlines()
        for ll in ls:
            id ,idx = ll.split('\t')
            dict_all[id] = idx
            if dict.has_key(id) is False:
                print 'www.freebase.com%s'%(id)
            else:
                count+=1

        print count