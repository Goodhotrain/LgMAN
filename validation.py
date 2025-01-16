import os
import time
import torch
from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy, compute_wa_f1
from tools.model import choose_save_checkpoint

def val_epoch(e, data_loader, model, criterion, opt, writer, optimizer):
    k, epoch, valid_acc = e
    print("# ---------------------------------------------------------------------- #")
    print('Validation at {}-fold epoch {}'.format(k, epoch))
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    end_time = time.time()
    
    all_preds = []
    all_labels = []
    for i, data_item in enumerate(data_loader):
        visual, target, audio,text, batch_size = process_data_item(opt, data_item)
        data_time.update(time.time() - end_time)
        with torch.no_grad():
            output, loss = run_model(opt, [visual, target, audio, text], model, criterion, i)

        acc, pre = calculate_accuracy(output, target)

        losses.update(loss.item(), batch_size)
        accuracies.update(acc, batch_size)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        all_preds.extend(pre.cpu().numpy().tolist())
        all_labels.extend(target.cpu().numpy().tolist())
    wa_f1_score = compute_wa_f1(all_labels, all_preds)
    
    writer.add_scalar(f'val/loss/{k}_fold', losses.avg, epoch)
    writer.add_scalar(f'val/acc/{k}_fold', accuracies.avg, epoch)
    print("Val loss: {:.4f}".format(losses.avg))
    print("Val acc: {:.4f}".format(accuracies.avg))
    print(f"Weighted Average F1 Score: {wa_f1_score:.4f}")

    # if not opt.debug:
    #     save_file_path = os.path.join(opt.ckpt_path, 'save_{}-fold_{}.pth'.format(k, epoch))
    #     states = {
    #         'epoch': epoch + 1,
    #         'state_dict': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #     }
    #     if epoch%2 == 0:
    #         torch.save(states, save_file_path)
    # if opt.mode == 'pretrain':
    #     if epoch%10==0:
    #         save_file_path = os.path.join(opt.ckpt_path, f'{k}_{epoch}model_state.pth')
    #         states = {
    #             'epoch': epoch + 1,
    #             'state_dict': choose_save_checkpoint(model),
    #             'optimizer': optimizer.state_dict(),
    #         }
    #         torch.save(states, save_file_path)
    # else :
    #     if epoch%10==0:
    #         save_file_path = os.path.join(opt.ckpt_path, f'{k}_{epoch}model_state.pth')
    #         states = {
    #             'epoch': epoch + 1,
    #             # 'state_dict': choose_save_checkpoint(model),
    #             'state_dict': model.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #         }
    #         torch.save(states, save_file_path)
    return max(accuracies.avg, valid_acc)