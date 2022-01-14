import tensorflow as tf
from tensorflow import keras
class residual_part(keras.layers.Layer):
    def __init__(self,filters=1024,keras_size=3,stride=1,is_begin=False):
        super(residual_part, self).__init__()
        self.cond1=keras.layers.Conv1D(filters=filters,kernel_size=keras_size,strides=stride)
        self.bn1=keras.layers.BatchNormalization()
        self.relu1=keras.layers.ReLU()
        self.drop1=keras.layers.Dropout(0.25)
        self.is_begin=is_begin
    def call(self, inputs, **kwargs):
        if self.is_begin:
            inputs=tf.reshape(inputs[:,:,:2],(243,34))
        x=self.cond1(inputs)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.drop1(x)
        return x
class residual(keras.layers.Layer):
    #is_conv参数代表是否用卷积代替切片操作
    def __init__(self,filters=1024,keras_size=3,is_residual=True,is_conv=False,stride=1):
        super(residual, self).__init__()
        self.residul_part1=residual_part(filters,keras_size,stride=stride)
        self.residul_part2=residual_part(filters,keras_size=1)
        self.residul_part3=residual_part(filters,keras_size=keras_size+2)
        self.is_residual=is_residual
        self.keras_size=keras_size
        self.is_conv=is_conv
    def call(self, inputs, **kwargs):
        x=self.residul_part1(inputs)
        x=self.residul_part2(x)
        if self.is_residual:
            if self.is_conv:
                y=self.residul_part3(inputs)
            else:
                y=inputs[:,self.keras_size-1::]
            x=y+x
        return x


model=keras.Sequential(
    [
        residual_part(is_begin=True),
        residual(keras_size=7),
        residual(keras_size=19),
        residual(keras_size=55),
        residual(keras_size=163),
        residual_part(filters=51,keras_size=1),
        keras.layers.Reshape((1,17,3))
    ]
)
if __name__ == '__main__':
    model=model
    model.build((None,243,34))
    model.summary()