import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


def main():
    #Columns name to 
    col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
        "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
        "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
        "num_shells","num_access_files","num_outbound_cmds","is_host_login",
        "is_guest_login","count","srv_count","serror_rate", "srv_serror_rate",
        "rerror_rate","srv_rerror_rate","same_srv_rate", "diff_srv_rate", "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
        "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate","attack", "last_flag"]

    test_df = pd.read_csv('Test.txt', columns = col_names)
    train_df = pd.read_csv('Train.txt', columns = col_names)

    train_X, train_y, train_le = preprocess_data(train_df)
    test_X, test_y, train_le = preprocess_data(test_df)
    

def preprocess_data(df):
    X = df.iloc[:, :-2]
    y = df.iloc[:,-2]
    
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y, le

def random_forest(X, y):


def gradient_boost(X, y):
    