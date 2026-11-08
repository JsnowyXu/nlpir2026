import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from utils.metric import get_metric
from models.dhpt import DHPT

def evaluation(dhpt, batches_dt, logger, device, string='validation'):
    dhpt.eval()
    with torch.no_grad():
        criterion = torch.nn.BCELoss()

        keys = ['group']
        results = {k: {} for k in keys}

        # Training loop. For each batch...
        total_sample_numbers, total_loss = 0, 0
        batch_range = tqdm(enumerate(batches_dt))
        for idx, batch_train in batch_range:
            # logger.info("this time is:{}".format(cott))
            # if idx > 1:
            #     break
            x, truths = batch_train
            gpu_word_text, texts, labels = [], [], []
            com_num,com_not_initial,com_label_feats,com_word_text_feats, com_bert_text_feats, \
            com_bert_text_masks, com_bert_text_token_type_ids,com_word_text_len,com_bert_text_len,com_label_len,\
            com_publication_date,com_ts, bert_text_input,bert_text_mask,bert_text_token_type_ids,word_inputs,publicationDate,ts = x

            bert_text_input = bert_text_input.to(device, dtype=torch.long)
            bert_text_mask = bert_text_mask.to(device, dtype=torch.long)
            bert_text_token_type_ids = bert_text_token_type_ids.to(device, dtype=torch.long)
            for elemt in word_inputs:
                gpu_word_text.append(elemt.to(device))
            gpu_word_text = pad_sequence(gpu_word_text, batch_first=True, padding_value=0).to(device)
            
            total_sample_numbers += len(bert_text_input)

            # gpu_x = pad_sequence(gpu_x, batch_first=True, padding_value=0).to(device)
            com_bert_text_token_type_ids = torch.stack(com_bert_text_token_type_ids)
            com_bert_text_feats = torch.stack(com_bert_text_feats)
            com_bert_text_masks = torch.stack(com_bert_text_masks)
            
            local_logits = dhpt(bert_text_input, bert_text_mask, bert_text_token_type_ids,gpu_word_text,
                               publicationDate,ts,{'text': com_word_text_feats,'com_date':com_publication_date,'ts':torch.Tensor(com_ts).to(device,dtype=torch.float), 'label': com_label_feats},
                                {'text': com_word_text_len,'com_num':com_num,'label': com_label_len}, com_not_initial)
            
            loss = 0
            truths = [truth.to(device) for truth in truths]
            for truth, pred in zip(truths, local_logits):
                loss += criterion(pred, truth)
            
            batch_total_loss = loss.cpu().data.numpy()
            total_loss += batch_total_loss

            for j, (truth, pred, key) in enumerate(zip(truths, local_logits, keys)):
                scores = get_metric(y_true=truth.detach().cpu(), y_pred=pred.detach().cpu(), idx=j)
                for metric_i, metric_value in scores.items():
                    if metric_i not in results[key]:
                        results[key][metric_i] = 0
                    results[key][metric_i] += metric_value
            
            batch_range.set_description(f"loss: {batch_total_loss},{batch_total_loss/len(bert_text_input)},{total_loss};")

        for k, k_items in results.items():
            for met_key, met_value in k_items.items():
                results[k][met_key] /= total_sample_numbers
        valloss = total_loss / total_sample_numbers
        logger.info("Loss in {}:{}".format(string, total_loss / total_sample_numbers))
        logger.info("Metrics in {}:{}".format(string, results))

        return results, valloss
