from torch.utils.tensorboard import SummaryWriter
import torch
from torch.optim import Adam
from embeddings import GloveEmbedding
import numpy as np


def train(model, train_dataloader, validation_dataloader, embedding, name="", num_epochs=100, lr=0.001):
    train_writer = SummaryWriter(log_dir=f'logs/{name}_train')
    val_writer = SummaryWriter(log_dir=f'logs/{name}_val')
    optimizer = Adam(model.parameters(), lr=lr)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    embedding = embedding.to(device)
    if device == torch.device('cuda'):
        model.cuda()

    for epoch in range(num_epochs):
        model.train()
        train_loss = []
        accuracy_num_train = []
        accuracy_train = []
        predictions_train = []
        for i, batch in enumerate(train_dataloader):

            optimizer.zero_grad()

            prediction = model.process_batch(batch, embedding)

            loss = model.loss(prediction, batch)
            loss.backward()

            accuracy = model.accuracy(prediction, batch)

            optimizer.step()

            train_loss += [loss.detach().cpu().numpy()]
            #some of the models returns two accuracies
            if isinstance(accuracy, tuple):
                accuracy_num, accuracy = accuracy
                prediction_num, prediction = prediction
                accuracy_num_train += [accuracy_num.detach().cpu().numpy()]

            accuracy_train += [accuracy]
            predictions_train += [prediction.detach().cpu().numpy()]

        train_writer.add_scalar('loss', np.mean(train_loss), epoch)
        train_writer.add_scalar('accuracy', np.mean(accuracy_train), epoch)
        #train_writer.add_histogram('predictions',predictions_train, epoch)

        if len(accuracy_num_train)>0:
            train_writer.add_scalar('accuracy_num', np.mean(accuracy_num_train), epoch)


        model.eval()
        val_loss = []
        accuracy_num_val = []
        accuracy_val = []
        predictions_val = []
        for batch in iter(validation_dataloader):

            prediction = model.process_batch(batch, embedding)

            accuracy = model.accuracy(prediction, batch)

            val_loss += [loss.detach().cpu().numpy()]

            if isinstance(accuracy, tuple):
                accuracy_num, accuracy = accuracy
                predictions_num, prediction = prediction
                accuracy_num_val += [accuracy_num.detach().cpu().numpy()]

            accuracy_val += [accuracy]
            predictions_val += [prediction.detach().cpu().numpy()]


        val_writer.add_scalar('loss', np.mean(val_loss), epoch)

        val_writer.add_scalar('accuracy', np.mean(accuracy_val), epoch)
        #train_writer.add_histogram('predictions',predictions_train, epoch)
        if len(accuracy_num_train)>0:
            val_writer.add_scalar('accuracy_num', np.mean(accuracy_num_val), epoch)

        print(f"EPOCH {epoch}")
        #print("training ")
        #print(f"loss = {np.mean(train_loss):.3f} accuracy_num = {np.mean(accuracy_num_train):.3f}  accuracy = {np.mean(accuracy_train):.3f}")
        #print("validation ")
        #print(f"loss = {np.mean(val_loss):.3f} accuracy_num = {np.mean(accuracy_num_val):.3f}  accuracy = {np.mean(accuracy_val):.3f}")


if __name__ == '__main__':
    from keyword_predictor import KeyWordPredictor
    from col_predictor import ColPredictor
    from andor_predictor import AndOrPredictor
    from agg_predictor import AggPredictor
    from op_predictor import OpPredictor
    from having_predictor import HavingPredictor
    from desasc_limit_predictor import DesAscLimitPredictor
    from utils.dataloader import SpiderDataset, try_tensor_collate_fn
    from embeddings import GloveEmbedding
    from torch.utils.data import DataLoader
    import argparse

    emb = GloveEmbedding(path='glove/glove.6B.50d.txt')
    spider_train = SpiderDataset(data_path='train.json', tables_path='tables.json', exclude_keywords=["between", "distinct", '-', ' / ', ' + '])
    spider_dev = SpiderDataset(data_path='dev.json', tables_path='tables.json', exclude_keywords=["between", "distinct", '-', ' / ', ' + '])
    
    choices=['column','keyword','andor','agg','op','having','desasc']
    models= [ColPredictor,KeyWordPredictor,AndOrPredictor,OpPredictor,HavingPredictor,DesAscLimitPredictor]
	
	
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epochs',  default=100, type=int)
    parser.add_argument('--batch_size', default=248, type=int)
    parser.add_argument('--name_postfix',default='', type=str)
    parser.add_argument('--use_gpu', default=False, type=bool)
    parser.add_argument('--N_word', default=50, type=int)
    parser.add_argument('--hidden_dim', default=30, type=int)
    parser.add_argument('--model', choices=['column','keyword','andor','agg','op','having','desasc'], default='having')
    args = parser.parse_args()
	
    #if args.model in choices:
    #    print('hi')
    #    model=models[choices.index(args.model)](N_word=args.N_word, hidden_dim=args.hidden_dim, num_layers=args.num_layers, gpu=args.use_gpu)
    #    train_set = spider_train.generate_column_dataset()
    #    validation_set = spider_dev.generate_column_dataset()

    if args.model == 'column':
        model = ColPredictor(N_word=50, hidden_dim=30, num_layers=args.num_layers, gpu=args.use_gpu)
        train_set = spider_train.generate_column_dataset()
        validation_set = spider_dev.generate_column_dataset()
    
    elif args.model == 'keyword':
        model = KeyWordPredictor(N_word=50, hidden_dim=30, num_layers=args.num_layers, gpu=args.use_gpu)
        train_set = spider_train.generate_keyword_dataset()
        validation_set = spider_dev.generate_keyword_dataset()
        
    elif args.model == 'andor':
        model = AndOrPredictor(N_word=50, hidden_dim=30, num_layers=args.num_layers, gpu=args.use_gpu)
        train_set = spider_train.generate_andor_dataset()
        validation_set = spider_dev.generate_andor_dataset()

    elif args.model == 'agg':
        model = AggPredictor(N_word=50, hidden_dim=30, num_layers=args.num_layers, gpu=args.use_gpu)
        train_set = spider_train.generate_agg_dataset()
        validation_set = spider_dev.generate_agg_dataset()
    
    elif args.model == 'op':
        model = OpPredictor(N_word=50, hidden_dim=30, num_layers=args.num_layers, gpu=args.use_gpu)
        train_set = spider_train.generate_op_dataset()
        validation_set = spider_dev.generate_op_dataset()
                        
    elif args.model == 'having':
        model = HavingPredictor(N_word=50, hidden_dim=30, num_layers=args.num_layers, gpu=args.use_gpu)
        train_set = spider_train.generate_having_dataset()
        validation_set = spider_dev.generate_having_dataset()

    elif args.model == 'desasc':
        model = DesAscLimitPredictor(N_word=50, hidden_dim=30, num_layers=args.num_layers, gpu=args.use_gpu)
        train_set = spider_train.generate_desasc_dataset()
        validation_set = spider_dev.generate_desasc_dataset()
        
        
    dl_train = DataLoader(train_set, batch_size=args.batch_size, collate_fn=try_tensor_collate_fn)
    dl_validation = DataLoader(validation_set, batch_size=len(validation_set), collate_fn=try_tensor_collate_fn)

    train(model, dl_train, dl_validation, emb, 
            name=f'{args.model}__num_layers={args.num_layers}__lr={args.lr}__{args.name_postfix}', 
            num_epochs=args.num_epochs,
            lr=args.lr)
