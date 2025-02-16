import torch
from dataset import get_data_transforms, load_data
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet50, de_wide_resnet50_2
from dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
from copy import copy
from new_model import FPA, DAM_layers
from fightingcv_attention.attention.ACmixAttention import ACmix
from gau_decoder import GAU_Decoder, Bottleneck

import os
import warnings
warnings.filterwarnings("ignore")


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def cal_anomaly_map_single_add(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = fs
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    # fs = fs_list[i]
    # ft = ft_list[i]
    fs = fs_list
    ft = ft_list
    #fs_norm = F.normalize(fs, p=2)
    #ft_norm = F.normalize(ft, p=2)
    a_map = 1 - F.cosine_similarity(fs, ft)
    # a_map = fs
    a_map = torch.unsqueeze(a_map, dim=1)
    a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
    a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
    if amap_mode == 'mul':
        anomaly_map *= a_map
    else:
        anomaly_map += a_map
    return anomaly_map,a_map_list


def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluation(encoder, bn, decoder,acm,fpn,att_layer, dataloader,device,_class_=None):
    #_, t_bn = resnet50(pretrained=True)
    #bn.load_state_dict(bn.state_dict())
    bn.eval()
    #bn.training = False
    #t_bn.to(device)
    #t_bn.load_state_dict(bn.state_dict())
    decoder.eval()

    acm.eval()
    fpn.eval()
    att_layer.eval()

    total_params_encoder = count_parameters(encoder)
    print(total_params_encoder)
    total_params_decoder = count_parameters(decoder)
    print(total_params_decoder)
    total_params_acm = count_parameters(acm)
    print(total_params_acm)
    total_params_fpn = count_parameters(fpn)
    print(total_params_fpn)
    total_params_att_layer = count_parameters(att_layer)
    print(total_params_att_layer)
    total_params_bn = count_parameters(bn)
    print(total_params_bn)

    total_params = (total_params_encoder + total_params_decoder + total_params_bn + total_params_acm + total_params_att_layer)

    print(f"Total Params: {total_params}")

    from fvcore.nn import FlopCountAnalysis

    # 定义输入张量
    input1 = torch.randn(1, 3, 256, 256).to(device)  # 根据模块的输入调整
    # input2 = torch.randn(512, 256, 3, 3).to(device)  # 根据模块的输入调整
    input3 = torch.randn(1, 2048, 8, 8).to(device)  # 根据模块的输入调整
    input4 = torch.randn(1, 2048, 8, 8).to(device)  # 根据模块的输入调整
    # input5 = torch.randn(1, 2048, 8, 8).to(device)  # 根据模块的输入调整
    input6 = torch.randn(1, 2048, 8, 8).to(device)  # 根据模块的输入调

    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    count = 1

    result_path = './results_all/' + _class_ + '/'
    if not os.path.exists(result_path): os.makedirs(result_path)

    with torch.no_grad():
        l = []
        for img, gt, label, _ in dataloader:
            # s = time()

            img = img.to(device)

            inputs = encoder(img)

            bn_outputs = bn(inputs)  # bn(inputs))

            acm_outputs = acm(bn_outputs)

            # import numpy as np
            import matplotlib.pyplot as plt
            from sklearn.metrics.pairwise import cosine_similarity
            import seaborn as sns
            from scipy.stats import kurtosis

            # 假设有两个四维张量A和B
            # up_ = torch.nn.Upsample(size=64, mode='bilinear')
            # bn_outputs = up_(bn_outputs)
            # acm_outputs_ = up_(acm_outputs)
            # print(bn_outputs.shape)
            # a = bn_outputs.cpu().squeeze().mean(dim=0).numpy()
            # b = acm_outputs_.cpu().squeeze().mean(dim=0).numpy()
            # c = inputs[2].cpu().squeeze().mean(dim=0).numpy()

            # a = bn_outputs.cpu().squeeze().flatten().numpy()
            # b = acm_outputs.cpu().squeeze().flatten().numpy()
            # c = fd.cpu().squeeze().flatten().numpy()

            # similarity = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))
            # print(similarity)

            # 创建散点图
            # plt.figure(figsize=(10, 8))
            # plt.scatter(b[0], b[1])
            # plt.xlabel('Feature a')
            # plt.ylabel('Feature b')
            # plt.title('Scatter plot between Feature a and Feature b')
            # plt.show()

            # plt.figure(figsize=(10, 8))
            # sns.kdeplot(similarity[0], label='Similarity')
            # # plt.legend()
            # plt.show()

            # sns.kdeplot(b, label='acm')  # 使用核密度估计可视化数据分布
            # sns.kdeplot(a, label='bn')  # 使用核密度估计可视化数据分布
            # plt.xlabel('Value')
            # plt.ylabel('Density')
            # plt.title('Data Distribution')
            # plt.show()

            # kurtosis_value = kurtosis(c)
            # l.append(str(f'{kurtosis_value:.3f}'))

            # plt.hist(b, bins=20)  # 使用20个箱子表示数据分布
            # plt.xlabel('Value')
            # plt.ylabel('Frequency')
            # plt.title('Data Distribution')
            # plt.text(0.5, 0.9, f'Kurtosis: {kurtosis_value:.3f}', ha='center', va='center', transform=plt.gca().transAxes)
            # plt.show()


            fpn_outputs = fpn(acm_outputs)

            input_attention = copy(inputs)
            input_attention = att_layer(input_attention)

            f_list = []
            # f_list.append(fpn_outputs)
            f_list.append(fpn_outputs)
            f_list += input_attention

            outputs = decoder(f_list)  # bn(inputs)

            # anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            # e = time()
            # t = e - s
            # print(t)


            ano_map = min_max_norm(anomaly_map)
            heatmap = cvt2heatmap(ano_map * 255)
            cv2.imwrite(result_path + str(count) + '_' + 'heatmap.png', heatmap)
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img) * 255)
            img_heatmap = show_cam_on_image(img, heatmap)
            cv2.imwrite(result_path + str(count) + '_' + '.png', img)
            cv2.imwrite(result_path + str(count) + '_' + 'img_heatmap.png', img_heatmap)

            # gt[gt > 0.5] = 1
            # gt[gt <= 0.5] = 0
            gt = gt.bool()
            if label.item()!=0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis,:,:]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

            gt = gt.cpu().numpy().astype(int)[0][0] * 255
            cv2.imwrite(result_path + str(count) + '_' + 'gt.png', gt)

            count += 1
        # print(l)

        #ano_score = (pr_list_sp - np.min(pr_list_sp)) / (np.max(pr_list_sp) - np.min(pr_list_sp))
        #vis_data = {}
        #vis_data['Anomaly Score'] = ano_score
        #vis_data['Ground Truth'] = np.array(gt_list_sp)
        # print(type(vis_data))
        # np.save('vis.npy',vis_data)
        #with open('{}_vis.pkl'.format(_class_), 'wb') as f:
        #    pickle.dump(vis_data, f, pickle.HIGHEST_PROTOCOL)


        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
    return auroc_px, auroc_sp, round(np.mean(aupro_list),3)

def test(_class_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(device)
    print(_class_)

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = './mvtec/' + _class_
    ckp_path_best = './checkpoints/' + 'wres50_' + _class_ + '_best.pth'
    ckp_path_last = './checkpoints/' + 'wres50_' + _class_ + '_last.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)

    acm = ACmix(in_planes=2048, out_planes=2048)
    acm = acm.to(device)

    fpn = FPA(channels=2048).to(device)
    att_layer = DAM_layers().to(device)

    encoder.eval()
    decoder = GAU_Decoder(block=Bottleneck)
    decoder = decoder.to(device)

    ckp = torch.load(ckp_path_best)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)

    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    acm.load_state_dict(ckp['acm'])
    fpn.load_state_dict(ckp['fpn'])
    att_layer.load_state_dict(ckp['att_layer'])

    auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder,acm,fpn,att_layer, test_dataloader, device,_class_)
    print(_class_,':',auroc_px,',',auroc_sp,',',aupro_px)
    return auroc_px, auroc_sp, aupro_px


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    items = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
     'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']
    result_list = []
    result_list_best = []


    for i in items:
        auroc_px, auroc_sp, aupro_px = test(i)
        result_list.append([i, auroc_px, auroc_sp, aupro_px])

    mean_auroc_px = np.mean([result[1] for result in result_list])
    mean_auroc_sp = np.mean([result[2] for result in result_list])
    mean_aupro_px = np.mean([result[3] for result in result_list])
    print(result_list)
    print('mPixel Auroc:{:.4f}, mSample Auroc:{:.4f}, mPixel Aupro:{:.4}'.format(mean_auroc_px, mean_auroc_sp,
                                                                                    mean_aupro_px))