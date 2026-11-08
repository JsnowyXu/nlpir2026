import argparse


def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give good results without cross-validation.
    """
    parser = argparse.ArgumentParser(description="Run patent labeling with hierarchy")

    # Data Parameters
    parser.add_argument("--data", default="IN", help="Dataset Name.")
    parser.add_argument("--gpu", type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument("--US", type=str, default='IN', help='dataset of US or CN')
    parser.add_argument("--use-bert", type=bool, default=True, help='use BERT or Word2vec for text embedding')
    parser.add_argument("--use-cnn", type=bool, default=True, help='use Text-CNN for further text embedding')
    parser.add_argument("--use-his-memory", type=bool, default=True, help='use his memory for company patent codes')
    parser.add_argument("--use-memory", type=bool, default=True, help='use memory')
    parser.add_argument("--use-company", type=bool, default=True, help='use historical patent sequences')
    parser.add_argument("--use-hierarchy", type=bool, default=True, help='use hierarchical prediction')
    parser.add_argument("--use-horizontal", type=bool, default=True, help='use GCN in decoder')
    parser.add_argument("--use-vertical", type=bool, default=True, help='fuse information vertically')
    parser.add_argument('--use_time', type=bool, default=True, help='Use time encoder embedding')
    parser.add_argument("--level", type=int, default=3, help="The class type.")

    parser.add_argument("--train-file", nargs="?",
                        default="/home/xuxj/PatentLabelling/patent-classification/PreProcess1/datas/{}-filter/new-{}/train-val-test/train.csv",
                        help="Training data.")
    parser.add_argument("--validation-file", nargs="?",
                        default="/home/xuxj/PatentLabelling/patent-classification/PreProcess1/datas/{}-filter/new-{}/train-val-test/validate.csv",
                        help="Validation data.")
    parser.add_argument("--test-file", nargs="?",
                        default="/home/xuxj/PatentLabelling/patent-classification/PreProcess1/datas/{}-filter/new-{}/train-val-test/test.csv",
                        help="Testing data.")
    parser.add_argument("--testdata-file", nargs="?",
                        default="/home/xuxj/PatentLabelling/patent-classification/PreProcess1/datas/{}-filter/new-{}/train-val-test/testdata.csv",
                        help="Testing data.")
    parser.add_argument("--metadata-file", nargs="?", default="../data/metadata.tsv",
                        help="Metadata file for embedding visualization.")
    parser.add_argument("--adjacent-dt", nargs="?",
                        default="/home/xuxj/PatentLabelling/patent-classification/PreProcess1/datas/{}-filter/adjacent/clsIII-reInd.npy",
                        help="adjacent matrix.")
    parser.add_argument("--hierarchy_tree", nargs="?",
                        default="/home/xuxj/PatentLabelling/patent-classification/PreProcess1/datas/{}-filter/map/tree/cls{}.csv",
                        help="hierarchy_tree")
    parser.add_argument("--hierarchy_names", type=list,
                        default=["I-II", "II-III"],
                        help="tree_names")
    # parser.add_argument("--hierarchy_names", type=list,
    #                     default=["I-II", "II-III", "III-IV"],
                        # help="tree_names")
    parser.add_argument("--word2vec-file", nargs="?", default="/home/xuxj/PatentLabelling/patent-classification/PreProcess1/NLPModel/word2vec-{}-kdd-200K.kv",
                        help="Word2vec file for embedding characters (the dim need to be the same as embedding dim).")
    parser.add_argument("--raw-node-features", nargs="?", default="/home/xuxj/PatentLabelling/patent-classification/KDD2024/memories/{}/raw_node_features.npy",
                        help="Word2vec file for embedding characters (the dim need to be the same as embedding dim).")
    parser.add_argument("--bert-base", nargs="?",
                        default="/home/xuxj/PatentLabelling/patent-classification/Baseline/Bert/bert-base-{}",
                        help="bert pretrain model.")
    parser.add_argument("--bert-lang", nargs="?",
                        default="uncased",
                        help="bert model.")
    # Model Hyperparameters
    parser.add_argument("--embedding-type", type=int, default=3, help="The embedding type.")
    parser.add_argument("--bert-embedding-dim", type=int, default=768, help="Dimensionality of character embedding.")
    parser.add_argument("--wordembedding-dim", type=int, default=100, help="Dimensionality of character embedding for word2vec.")
    parser.add_argument("--memory-dim", type=int, default=100, help="Dimensionality of character embedding for word2vec.")
    parser.add_argument("--message-dim", type=int, default=100, help="Dimensionality of character embedding for word2vec.")
    parser.add_argument("--message-agg", type=str, default='last', help="Aggregator type for message.")
    parser.add_argument('--node_embedding_module', type=str, default="identity", choices=[
    "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
    parser.add_argument('--message_function', type=str, default="identity", choices=[
    "mlp", "identity"], help='Type of message function')
    parser.add_argument('--memory_updater', type=str, default="gru", choices=[
    "gru", "rnn"], help='Type of memory updater')
    parser.add_argument('--memory_update_at_end', action='store_true',
                        help='Whether to update memory at the end or at the start of the batch')
                    
    parser.add_argument("--lstm-dim", type=int, default=256, help="Dimensionality of LSTM neurons.")
    parser.add_argument("--fc-dim", type=int, default=256, help="Dimensionality for FC neurons.")
    parser.add_argument("--gcn_layer", type=int, default=2, help="Numbers for gcn layers")

    parser.add_argument("--fused_way", type=str, default='readout', help="Ways for gcn results:[last or readout]")
    parser.add_argument("--dropout-rate", type=float, default=0.3, help="Dropout keep probability.")
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    # parser.add_argument('--fuse_layer', type=int, default=1, help='fuse embedding from 1-3 layers')
    parser.add_argument('--use_mean', type=bool, default=False, help='Use mean function as prediction')
    parser.add_argument('--use_lstm_text', type=bool, default=True, help='Use bi-lstm function in text embedding')
    parser.add_argument('--model_way', type=str, default='lstm', help='Way of aggregation in historical behavior modelling')
    parser.add_argument('--com_feat_type', type=str, default='label', help='[text,time or label]')
    parser.add_argument("--max_len_bert", type=int, default=350, help="Number of characters for bert.")
    parser.add_argument("--max_len_word", type=int, default=350, help="Number of characters for word2vec.")
    parser.add_argument("--max_len_neighbors", type=int, default=30, help="Number of historical patents for a company .")
    parser.add_argument("--weight", type=int, default=0.9, help="The weight for historical embedding.")

    # bert_cnn
    parser.add_argument("--num-filters", type=int, default=256, help="Number of characters.")
    parser.add_argument("--filter-sizes", type=int, default=(2, 3, 4), help="Number of characters.")
    # Training Parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch Size.")
    parser.add_argument("--learning-rate", type=float, default=0.00003, help="Learning rate.")
    parser.add_argument("--decay-rate", type=float, default=0.95, help="Rate of decay for learning rate.")
    parser.add_argument("--decay-steps", type=int, default=100, help="How many steps before decay learning rate.")

    return parser.parse_args()
