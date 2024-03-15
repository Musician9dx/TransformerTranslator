import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Embedding, Add
from tensorflow import GradientTape
from tensorflow.keras import Model
from tensorflow.keras.activations import relu,softmax

class SelfAttention(Layer):

    def __init__(self, embedSize):
        super().__init__()
        self.embedSize = embedSize
        self.key = Dense(self.embedSize, activation="linear")
        self.query = Dense(self.embedSize, activation="linear")
        self.ad = Add()

    def call(self, logits):
        inputs = logits

        key = self.key(logits)
        query = self.query(logits)

        key = tf.matmul(query, key, transpose_b=True)

        value = tf.matmul(key, query)
        value = softmax(value)

        return value


class CrossAttention(Layer):

    def __init__(self, embedSize):
        super().__init__()
        self.embedSize = embedSize
        self.key = Dense(self.embedSize, activation="linear")
        self.query = Dense(self.embedSize, activation="linear")
        self.ad = Add()

    def call(self, logits1, logits2):
        key = self.key(logits1)
        query = self.query(logits2)

        key = tf.matmul(query, key, transpose_b=True)
        value = tf.matmul(key, query)
        value = softmax(value)

        return value


class MultiHeadAttention(Layer):

    def __init__(self, numHeads, embedSize):
        super().__init__()

        headSize = embedSize // numHeads

        self.attentionBlocks = [SelfAttention(headSize) for i in range(numHeads)]

    def call(self, logits):
        c = []

        for layer in self.attentionBlocks:
            op1 = layer(logits)
            c.append(op1)

        op2 = tf.concat(c, axis=-1)

        return op2


class Encoder(Layer):

    def __init__(self, vocabSize, embedSize):
        super().__init__()

        self.vocabSize = vocabSize
        self.embedSize = embedSize
        self.embedding = Embedding(self.vocabSize, self.embedSize)
        self.self_attention = SelfAttention(self.embedSize)
        self.cross_attention = CrossAttention(self.embedSize)
        self.mha = MultiHeadAttention(4, 100)
        self.pEmbedding=PositionEmbedding(vocabSize,embedSize)


    def call(self, tokens):
        embedding = self.embedding(tokens)
        embedding=embedding+self.pEmbedding(embedding)
        logits = self.mha(embedding)
        logits = self.self_attention(logits)
        logits = self.cross_attention(logits, logits)

        return logits


class Decoder(Layer):

    def __init__(self, vocabSize, embedSize):
        super().__init__()

        self.vocabSize = vocabSize
        self.embedSize = embedSize
        self.embedding = Embedding(self.vocabSize, self.embedSize)
        self.self_attention = SelfAttention(self.embedSize)
        self.mha = MultiHeadAttention(4, 100)
        self.maskMatrix = tf.ones((self.embedSize, self.embedSize))
        self.maskMatrix = tf.linalg.band_part(self.maskMatrix, -1, 0)
        self.pEmbedding=PositionEmbedding(vocabSize,embedSize)


    def call(self, tokens):
        embedding = self.embedding(tokens)
        embedding=embedding+self.pEmbedding(embedding)
        logits = self.mha(embedding)
        logits = self.self_attention(logits)
        logits = tf.matmul(logits, self.maskMatrix)
        return logits


class TransformerModel(Model):

    def __init__(self, vocabSize=5, embedSize=100):
        super().__init__()

        self.enc = Encoder(vocabSize, embedSize)
        self.dec = Decoder(vocabSize, embedSize)
        self.ca = CrossAttention(embedSize)
        self.layerAttention = [SelfAttention(embedSize) for i in range(4)]
        self.ad = Add()
        self.d1 = [Dense(embedSize, activation="relu") for i in range(5)]

    def call(self, encInputs, decInputs, targets=None):

        logits1 = self.enc(encInputs)
        logits2 = self.dec(decInputs)

        logits = self.ca(logits1, logits2)

        for layer in self.layerAttention:
            lg1 = logits
            logits = layer(logits)
            logits = self.ad([relu(lg1), logits])

        for layer in self.d1:
            logits = layer(logits)

        if targets is None:
            return logits, None
        else:
            B, T, C = tf.shape(logits)
            logits = tf.reshape(logits, (B * T, C))
            targets = tf.reshape(targets, (B * T,))
            loss = tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)
            loss = tf.reduce_mean(loss)
            return logits, loss

    #     def generate(self, idx,new_tokens):
    #         for _ in range(new_tokens):
    #             logits, _ = self.call(idx)
    #             logits = logits[:, -1, :]
    #             probs = softmax(logits, axis=-1)
    #             idx_next = tf.reshape(tf.argmax(probs, axis=-1), (-1, 1))
    #             idx = tf.concat([idx, tf.cast(idx_next, tf.int32)], axis=1)
    #         return idx

    def fitM(self, xb, yb, targets, steps=100):
        optimizer = tf.keras.optimizers.Adam()

        for step in range(steps):
            with tf.GradientTape() as tape:
                logits, loss = self.call(xb, yb, targets)
                print(f"Step: {step}", float(loss))
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))

