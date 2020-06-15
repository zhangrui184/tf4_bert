import tensorflow as tf
import nice_model as nice_model
import numpy as np
# ---------------------------------------------------------------------------------------------------------------------------------------
batch_size = 5


def source_to_seq(text):
    '''
    对源数据进行转换
    '''
    sequence_length = 120
    return [nice_model.source_letter_to_int.get(word, nice_model.source_letter_to_int['_UNK']) for word in
            text.split(' ')] + [nice_model.source_letter_to_int['_PAD']] * (sequence_length - len(text))


# 测试
# 输入一个单词
# input_word=[]
# with open("D:/python project me/data\text_summar\nice data\nice data dev/content_dev_id.txt") as f2:
#	for line in f2:
#	input_line=line
#	input_word

input_word = "国际 显示 人员 希望 数据 告诉 媒体 一定 关注 近日 调查"
# input_word = "3288 117 492 7 234 1338 3621 475 23 4 20 7 234 475 5 7 48 1577 369 1042 297 1924 1083 6 8 411 94 59 39 3095 11 4169 11 2616 11 2765 17 3068 216 127 274 3199 42 7 234 2456 320 8028 475 6"
'''
 NUMBER   反对 宪法 基本 原则 危害 国家 安全 政权 稳定 统一 的 煽动 民族 仇恨 民族 歧视 的   NUMBER   宣扬 邪教 和 封建迷信 的 散布 谣言 破坏 社会 稳定 的 侮辱 诽谤 他人 侵害 他人 合法权益 的   NUMBER   散布 淫秽 色情 赌博 暴力 凶杀 恐怖 或者 教唆 犯罪 的
 '''
text = source_to_seq(input_word)

checkpoint = "./checkpoint_f/trained_model.ckpt"

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # 加载模型
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')

    answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                      target_sequence_length: [len(input_word)] * batch_size,
                                      source_sequence_length: [len(input_word)] * batch_size})[0]

pad = nice_model.source_letter_to_int["_PAD"]
np.savetxt('D:\python project me\data/text_summar/nice data/result/answer_logits.txt',answer_logits)#answer_logits的ndarray,同l1=np.arange(5)

line1=[]#换行相连的数字
with open("D:\python project me\data/text_summar/nice data/result/answer_logits.txt","r",encoding="utf-8") as f1:#answer_logits.txt是目标的数字，换行相连
    for line in f1.readlines():
        line1.append(line)
line2=" ".join(i.replace("\n","") for i in line1)#空格相连的数字
with open("D:\python project me\data/text_summar/nice data/result/answer_logits_id.txt","w") as f2:#answer_logits_id.txt是目标的数字，空格相连
    f2.write(line2)


line3=[nice_model.source_int_to_letter[word] for word in answer_logits] #对应的字，list存储，每个是str
line4="".join(i for i in line3) #没有间隔，把line3的每个str存储到一行
with open("D:\python project me\data/text_summar/nice data/result/answer_logits_word.txt","w") as f2:#answer_logits_word.txt是目标的汉字
    f2.write(line4)
print('原始输入:', input_word)
#with open("D:\python project me\data/text_summar\conclusion\conclusion.txt","r",encoding="utf-8") as f1:
  # con=[]

  #  f1.write(str([line for line in answer_logits if line !=pad]))
#with open("D:\python project me\data/text_summar\conclusion\conclusion_id.txt","r",encoding="utf-8") as f1:
 #   f1.write(str("".join([nice_model.target_int_to_letter[line] for line in answer_logits if line !=pad])))
"""
print('\nSource')
print('  Word 编号:    {}'.format([i for i in text]))
print('  Input Words: {}'.format(" ".join([nice_model.source_int_to_letter[i] for i in text])))

print('\nTarget')
print('  Word 编号:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format(" ".join([nice_model.target_int_to_letter[i] for i in answer_logits if i != pad])))
"""