import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Mammogram Classification Model")

    # Training Params
    parser.add_argument('--checkpoint_model_save', 
                        type=str, 
                        default='/home/cse/visitor/abansal.visitor/scratch/focalnet_dino/other_approaches/resnext50/models/bin/model_image.pt',
                        help='Path to save checkpoint for text model')
    parser.add_argument('--file_path', 
                        type=str, 
                        default='/home/cse/visitor/abansal.visitor/scratch/focalnet_dino/other_approaches/resnext50/models/bin/training_stats.txt',
                        help='Path to save training statistics')
    parser.add_argument('--plot_path', 
                        type=str, 
                        default='/home/cse/visitor/abansal.visitor/scratch/focalnet_dino/other_approaches/resnext50/models/bin/loss_plot.png',
                        help='Path to save loss plot')
    parser.add_argument('--num_epochs', 
                        type=int, 
                        default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default=5e-6,
                        help='Learning rate for the optimizer')

    # DataLoader Params
    parser.add_argument('--topk', 
                        type=int, 
                        default=32,
                        help='Top-k parameter for DataLoader')
    parser.add_argument('--num_workers', 
                        type=int, 
                        default=8,
                        help='Number of DataLoader workers')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=64,
                        help='Batch size for training')

    # Model Params
    parser.add_argument('--layers_freeze', 
                        type=int, 
                        default=2,
                        help='Number of layers to freeze in the model')
    parser.add_argument('--img_size', 
                        type=int, 
                        default=384,
                        help='Input image size for the model')
    parser.add_argument('--model_type', 
                        type=str, 
                        default="vitb16_clip",
                        required=True,
                        help='Choose foundational model to train')
    parser.add_argument('--dataset_source', 
                        type=str,
                        help='Choose the source dataset you are working with')
    parser.add_argument('--dataset_target', 
                        type=str,
                        help='Choose the target dataset you are working with')
    parser.add_argument('--class_num', 
                        type=str,
                        required=True,
                        help='Number of classes')

    args = parser.parse_args()
    return args



