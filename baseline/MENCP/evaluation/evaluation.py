import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from utils.metric import get_metric


def evaluation(harnn, batches_dt, logger, device, use_XML=False, string='validation'):
    harnn.eval()
    with torch.no_grad():
        criterion = torch.nn.BCELoss()

        keys = ['group']
        results = {k: {} for k in keys}

        # Training loop. For each batch...
        cott = 0
        total_loss = 0
        batch_range = tqdm(enumerate(batches_dt))
        for idx, batch_train in batch_range:
            # logger.info("this time is:{}".format(cott))
            # if idx > 1:
            #     break

            x, truths = batch_train
            gpu_x, texts, labels = [], [], []
            companies, label_feats,word_text_feats, text_feats, text_masks, text_token_type_ids,word_text_len, text_len, label_len, not_initial, input,mask,token_type_ids,word_inputs = x

            input = input.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)

            for elemt in word_inputs:
                gpu_x.append(elemt.to(device))
            gpu_x = pad_sequence(gpu_x, batch_first=True, padding_value=0).to(device)

            cott += len(input)

            # gpu_x = pad_sequence(gpu_x, batch_first=True, padding_value=0).to(device)
            text_feats = torch.stack(text_feats)
            text_masks = torch.stack(text_masks)
            text_token_type_ids = torch.stack(text_token_type_ids)
            
            local_logits = harnn(input, mask, token_type_ids,gpu_x, 
                                        {'text': text_feats.to(device, dtype=torch.long),'mask':text_masks.to(device, dtype=torch.long), 'token_type_ids': text_token_type_ids.to(device, dtype=torch.long),'word_text':word_text_feats, 'label': label_feats},
                                        {'text': text_len,'word_text':word_text_len, 'label': label_len}, not_initial)

            loss = 0
            truths = [truth.to(device) for truth in truths]
            # print(truths[0].shape)
            for truth, pred in zip(truths, local_logits):
                # value, indices = truth.topk(5)
                # print("ground truth are: ", indices[:5])
                loss += criterion(pred, truth)
            
            batch_total_loss = loss.cpu().data.numpy()
            total_loss += loss.cpu().data.numpy()
            # print(loss.cpu().data.numpy())

            for j, (truth, pred, key) in enumerate(zip(truths, local_logits, keys)):
                scores = get_metric(y_true=truth.detach().cpu(), y_pred=pred.detach().cpu(), idx=j)
                for metric_i, metric_value in scores.items():
                    if metric_i not in results[key]:
                        results[key][metric_i] = 0
                    results[key][metric_i] += metric_value
            
            batch_range.set_description(f"loss: {batch_total_loss},{batch_total_loss/len(input)},{total_loss};")

        for k, k_items in results.items():
            for met_key, met_value in k_items.items():
                results[k][met_key] /= cott
        valloss = total_loss / cott
        logger.info("Loss in {}:{}".format(string, total_loss / cott))
        logger.info("Metrics in {}:{}".format(string, results))

        return results, valloss
