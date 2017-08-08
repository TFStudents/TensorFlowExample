#user/bin
import  tensorflow as tf

def imageTest():
    image_filename = "start_person.jpg"# 智能星人的图标
    # 获取图片数据
    file = open(image_filename, 'rb')
    data = file.read()
    file.close()
    # ---
    file_img = tf.image.decode_jpeg(data,channels=3)
    print(file_img)
    file_img=tf.expand_dims(file_img,0)
    print(file_img)
    with tf.Session() as sess:
        write = tf.summary.FileWriter("./test_2", graph=sess.graph)
        img = tf.summary.image("./", tensor=file_img)
        img_string = sess.run(img)
        write.add_summary(img_string)
imageTest()