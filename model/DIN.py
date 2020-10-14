import tensorflow as tf
from tensorflow.keras import layers
from layers.AUGRU import AUGRU
from layers.attention import attention
from layers.Dice import Dice, dice
from layers.AuxLayer import AuxLayer
import utils

class DIN(tf.keras.Model):
    def __init__(self, embedding_count_dict, embedding_dim_dict, embedding_features_list, user_behavior_features, activation="PReLU"):
        super(DIN, self).__init__(embedding_count_dict, embedding_dim_dict, embedding_features_list, user_behavior_features, activation)
        #Init Embedding Layer
        self.embedding_dim_dict = embedding_dim_dict
        self.embedding_count_dict = embedding_count_dict
        self.embedding_layers = dict()
        for feature in embedding_features_list:
            self.embedding_layers[feature] = layers.Embedding(embedding_count_dict[feature], embedding_dim_dict[feature])
        #DIN Attention+Sum pooling
        self.hist_at = attention(utils.get_input_dim(embedding_dim_dict, user_behavior_features))
        #Init Fully Connection Layer
        self.fc = tf.keras.Sequential()
        self.fc.add(layers.BatchNormalization())
        self.fc.add(layers.Dense(200, activation="relu")) 
        if activation == "Dice":
            self.fc.add(Dice())
        elif activation == "dice":
            self.fc.add(dice(200))
        elif activation == "PReLU":
            self.fc.add(layers.PReLU(alpha_initializer='zeros', weights=None))
        self.fc.add(layers.Dense(80, activation="relu"))
        if activation == "Dice":
            self.fc.add(Dice()) 
        elif activation == "dice":
            self.fc.add(dice(80))
        elif activation == "PReLU":
            self.fc.add(layers.PReLU(alpha_initializer='zeros', weights=None))
        self.fc.add(layers.Dense(2, activation=None))

    def get_emb_din(self, user_profile_dict, user_profile_list, hist_behavior_dict, target_item_dict, user_behavior_list):
        user_profile_feature_embedding = dict()
        for feature in user_profile_list:
            data = user_profile_dict[feature]
            embedding_layer = self.embedding_layers[feature]
            user_profile_feature_embedding[feature] = embedding_layer(data)
        
        target_item_feature_embedding = dict()
        for feature in user_behavior_list:
            data = target_item_dict[feature]
            embedding_layer = self.embedding_layers[feature]
            target_item_feature_embedding[feature] = embedding_layer(data)
        
        hist_behavior_embedding = dict()
        for feature in user_behavior_list:
            data = hist_behavior_dict[feature]
            embedding_layer = self.embedding_layers[feature]
            hist_behavior_embedding[feature] = embedding_layer(data)
        
        return utils.concat_features(user_profile_feature_embedding), utils.concat_features(target_item_feature_embedding), utils.concat_features(hist_behavior_embedding)
    
    def call(self, user_profile_dict, user_profile_list, hist_behavior_dict, target_item_dict, user_behavior_list, length):
        #Embedding Layer
        user_profile_embedding, target_item_embedding, hist_behavior_emebedding = self.get_emb_din(user_profile_dict, user_profile_list, hist_behavior_dict, target_item_dict, user_behavior_list)
        hist_attn_emb = self.hist_at(target_item_embedding, hist_behavior_emebedding, length)
        join_emb = tf.concat([user_profile_embedding, target_item_embedding, hist_attn_emb], -1)
        logit = tf.squeeze(self.fc(join_emb))
        output = tf.keras.activations.softmax(logit)
        return output, logit

if __name__ == "__main__":
    model = DIN(dict(), dict(), list(), list())