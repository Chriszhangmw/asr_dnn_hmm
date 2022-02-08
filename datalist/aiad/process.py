import pypinyin


a = "我喜欢吃火锅"

print(pypinyin.lazy_pinyin(a,style=pypinyin.Style.TONE3))


wav = open('./test.wav.lst','w',encoding='utf-8')
txt = open('./test.syllable.txt','w',encoding='utf-8')

with open('./test.csv','r',encoding='utf-8') as f:
    data = f.readlines()
    f.close()
count = 0
for line in data:
    line = line.strip().split(',')
    assert len(line) == 2
    path = line[0]
    text = line[1]
    s_text = pypinyin.lazy_pinyin(text,style=pypinyin.Style.TONE3)
    index = "Test" + str(count)
    wav.write(index + ' ' + path + '\n')
    txt.write(index + ' ' + ' '.join(s_text) + '\n')
    count +=1












