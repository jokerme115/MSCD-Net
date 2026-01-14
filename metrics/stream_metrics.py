import numpy as np
from sklearn.metrics import confusion_matrix

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k not in ["Class IoU", "Class Dice", "Class F1-Score", "Class Sensitivity", "Class Specificity", "Class Precision", "Class FP Rate", "Confusion Matrix"]:
                if isinstance(v, (int, float)):
                    string += f"{k}: {v:.4f}\n"
                else:
                    string += f"{k}: {v}\n"
        
        # 添加平均指标显示
        if "Class IoU" in results:
            mean_iou = np.mean(list(results["Class IoU"].values()))
            string += f"Mean IoU: {mean_iou:.4f}\n"
            
        if "Class Dice" in results:
            mean_dice = np.mean(list(results["Class Dice"].values()))
            string += f"Mean Dice: {mean_dice:.4f}\n"
            
        if "Class F1-Score" in results:
            mean_f1 = np.mean(list(results["Class F1-Score"].values()))
            string += f"Mean F1-Score: {mean_f1:.4f}\n"
            
        if "Class Sensitivity" in results:
            mean_sens = np.mean(list(results["Class Sensitivity"].values()))
            string += f"Mean Sensitivity: {mean_sens:.4f}\n"
            
        if "Class Specificity" in results:
            mean_spec = np.mean(list(results["Class Specificity"].values()))
            string += f"Mean Specificity: {mean_spec:.4f}\n"
            
        # 显示每个类别的指标
        string += "\nClass-wise Results:\n"
        if "Class IoU" in results:
            string += "Class IoU:\n"
            for k, v in results["Class IoU"].items():
                string += f"\tclass {k}: {v:.4f}\n"
                
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        # 修复除零错误
        acc_cls = np.divide(np.diag(hist), hist.sum(axis=1), out=np.zeros_like(np.diag(hist), dtype=float), where=hist.sum(axis=1)!=0)
        acc_cls = np.nanmean(acc_cls)
        # 修复除零错误
        iu = np.divide(np.diag(hist), (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)), 
                       out=np.zeros_like(np.diag(hist), dtype=float), 
                       where=(hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))!=0)
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        # 计算Dice系数，修复除零错误
        dice = np.divide(2 * np.diag(hist), (hist.sum(axis=0) + hist.sum(axis=1)), 
                         out=np.zeros_like(np.diag(hist), dtype=float), 
                         where=(hist.sum(axis=0) + hist.sum(axis=1))!=0)
        cls_dice = dict(zip(range(self.n_classes), dice))
        
        # 计算敏感度(Sensitivity/Recall)，修复除零错误
        sensitivity = np.divide(np.diag(hist), hist.sum(axis=1), 
                                out=np.zeros_like(np.diag(hist), dtype=float), 
                                where=hist.sum(axis=1)!=0)
        cls_sensitivity = dict(zip(range(self.n_classes), sensitivity))
        
        # 计算特异度(Specificity)
        # 对于多类问题，特异度是 (TN / (TN + FP))
        specificity = []
        for i in range(self.n_classes):
            # 对于类别i，TN是除了类别i之外的所有元素的对角线元素之和
            # FP是类别i在其他预测类别中的总和
            tn = np.sum(np.delete(np.diag(hist), i))
            fp = np.sum(np.delete(hist[:, i], i))
            if (tn + fp) > 0:
                specificity.append(tn / (tn + fp))
            else:
                specificity.append(0)  # 如果没有负样本，特异度为0
        cls_specificity = dict(zip(range(self.n_classes), specificity))
        
        # 计算Precision
        precision = np.divide(np.diag(hist), hist.sum(axis=0), out=np.zeros_like(np.diag(hist), dtype=float), where=hist.sum(axis=0)!=0)
        cls_precision = dict(zip(range(self.n_classes), precision))
        
        # 计算F1分数
        f1_score = np.zeros_like(precision)
        denominator = (precision + sensitivity)
        mask = denominator > 0
        f1_score[mask] = 2 * (precision * sensitivity)[mask] / denominator[mask]
        cls_f1_score = dict(zip(range(self.n_classes), f1_score))
        
        # 构建混淆矩阵详细信息
        confusion_details = {}
        for i in range(self.n_classes):
            tp = hist[i, i]  # 真正例
            fp = hist[:, i].sum() - tp  # 假正例
            fn = hist[i, :].sum() - tp  # 假负例
            tn = hist.sum() - tp - fp - fn  # 真负例
            
            confusion_details[i] = {
                'TP': tp,
                'FP': fp,
                'TN': tn,
                'FN': fn
            }

        # 计算假阳性率 (False Positive Rate)
        fp_rate = {}
        for i in range(self.n_classes):
            # 对于类别i，假阳性是其他类别被错误预测为类别i的数量
            fp = hist[:, i].sum() - hist[i, i]  # 所有非i类被预测为i类的总和
            tn = hist.sum() - hist[i, :].sum() - fp  # 真负例
            fp_rate[i] = fp / (fp + tn + 1e-10)  # 添加平滑项避免除零

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
                "Class Dice": cls_dice,
                "Class Sensitivity": cls_sensitivity,
                "Class Specificity": cls_specificity,
                "Class F1-Score": cls_f1_score,
                "Class Precision": cls_precision,
                "Class FP Rate": fp_rate,
                "Confusion Matrix": confusion_details
        }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg