import os
import math
import torch
import numpy as np

"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def log_average_miss_rate(precision, fp_cumsum, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.

        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    # if there were no detections of that class
    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num = 9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi

def calculate_mAP(tubes_dic, gt_tubes_dic, min_overlap):

    gt_counter_per_class = {}
    counter_videos_per_class = {}
    gt_files = {}

    for file_name in gt_tubes_dic.keys():

        gt_bounding_boxes = []
        already_seen_classes = []
        gt_tubes_ = gt_tubes_dic[file_name]

        for tub in gt_tubes_:

            class_id = str(tub[0].tolist())

            tube = tub[1:].view(-1,4).contiguous()
            gt_bounding_boxes.append({"class_id":class_id, "tube":tube, "used":False})
 
            if class_id in gt_counter_per_class:
                gt_counter_per_class[class_id] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[class_id] = 1

            if class_id not in already_seen_classes:
                if class_id in counter_videos_per_class:
                    counter_videos_per_class[class_id] += 1
                else:
                    # if class didn't exist yet
                    counter_videos_per_class[class_id] = 1
                already_seen_classes.append(class_id)

        gt_files[file_name] = gt_bounding_boxes

    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    detected_bboxes = {}

    for class_index, class_id in enumerate(gt_classes):

        bounding_boxes = []

        for file_name in tubes_dic:

            dt_tubes_ = tubes_dic[file_name]

            for tub in dt_tubes_:

                tmp_class_id = str(tub[0].tolist())
                confidence = tub[1]
                tube = tub[2:].view(-1,4).contiguous()

                if tmp_class_id == class_id:
                    bounding_boxes.append({"confidence":confidence, "file_id":file_name, "tube":tube})

        # sort detection-results by decreasing confidence
        bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)

        detected_bboxes[class_id] = bounding_boxes

    """
     Calculate the AP for each class
    """
    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}

    count_true_positives = {}

    results_file = open('results.txt', 'w')
    for class_index, class_id in enumerate(gt_classes):

        count_true_positives[class_id] = 0
        """
        Load detection-results of that class
        """
        dr_data = detected_bboxes[class_id]

        """
        Assign detection-results to ground-truth objects
        """
        nd = len(dr_data)
        tp = [0] * nd # creates an array of zeros of size nd
        fp = [0] * nd

        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]

            # assign detection-results to ground truth object if any
            ground_truth_data = gt_files[file_id]
            ovmax = -1
            gt_match = -1

            # load detected object bounding-box
            dt_tube =  detection["tube"] 

            for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_id"] == class_id:

                    gt_tube =  obj["tube"]
                    zero_area = ((dt_tube.eq(0).all(dim=1) == 1) & (gt_tube.eq(0).all(dim=1) == 1))
                    empty_indices = zero_area.nonzero().view(-1)

                    bi = [torch.max(dt_tube[:,0],gt_tube[:,0]), torch.max(dt_tube[:,1],gt_tube[:,1]), \
                          torch.min(dt_tube[:,2],gt_tube[:,2]), torch.min(dt_tube[:,3],gt_tube[:,3])]
                    bi = torch.stack(bi,dim=1)

                    iw = (bi[:,2] - bi[:,0] + 1).clamp_(min=0)
                    ih = (bi[:,3] - bi[:,1] + 1).clamp_(min=0)

                    ua = (dt_tube[:,2] - dt_tube[:,0] + 1) * (dt_tube[:,3] - dt_tube[:,1] + 1) + (gt_tube[:,2] - gt_tube[:,0]
                            + 1) * (gt_tube[:,3] - gt_tube[:,1] + 1) - iw * ih

                    ov = iw * ih / ua
                    ov.masked_fill_(zero_area,0)

                    ov = ov.sum()/ (dt_tube.size(0) - empty_indices.nelement())

                    if ov > ovmax:
                        ovmax = ov
                        gt_match = obj
            if ovmax >= min_overlap:

                if not bool(gt_match["used"]):
                    # true positive
                    tp[idx] = 1
                    gt_match["used"] = True
                    count_true_positives[class_id] += 1

                else:

                    fp[idx] = 1

            else:

                fp[idx] = 1
                if ovmax > 0:
                    status = "INSUFFICIENT OVERLAP"


        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        #print(tp)
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_id]
        #print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        #print(prec)

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap
        
        text = "{0:.2f}%".format(ap*100) + " = " + str(class_id) + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)
        rounded_prec = [ '%.2f' % elem for elem in prec ]
        rounded_rec = [ '%.2f' % elem for elem in rec ]

        results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
        print(text)

        n_videos = counter_videos_per_class[class_id]
        lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_videos)
        lamr_dictionary[class_id] = lamr
    results_file.write("\n# mAP of all classes\n")
    mAP = sum_AP / n_classes
    text = "mAP = {0:.2f}%".format(mAP*100)
    results_file.write(text + "\n")
    print(text)

    # iterate through all the files

    det_counter_per_class = {}
    for txt_file in tubes_dic.keys():
        # get lines to list
        tubes_list = tubes_dic[txt_file]
        for idx, tub in enumerate(tubes_list):
            class_id = str(tub[0].tolist())

            # count that object
            if class_id in det_counter_per_class:
                det_counter_per_class[class_id] += 1
            else:
                # if class didn't exist yet
                det_counter_per_class[class_id] = 1


    dr_classes = list(det_counter_per_class.keys())

    with open( "./results.txt", 'a') as results_file:
        results_file.write("\n# Number of ground-truth objects per class\n")
        for class_name in sorted(gt_counter_per_class):
            results_file.write(str(class_name) + ": " + str(gt_counter_per_class[class_name]) + "\n")

    with open( "./results.txt", 'a') as results_file:
        results_file.write("\n# Number of detected objects per class\n")
        print()
        for class_name in sorted(dr_classes):
            if class_name == 0:
                continue
            n_det = det_counter_per_class[class_name]
            text = str(class_name) + ": " + str(n_det)
            if class_name in count_true_positives:
                text += " (tp:" + str(count_true_positives[class_name]) + ""
                text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
            else:
                text += " (tp:" + str(0) + ""
                text += ", fp:" + str(n_det ) + ")\n"
            print(text)
            results_file.write(text)


if __name__ == '__main__':

    t = torch.Tensor([[18.0, 1.1856203079223633, 43.0, 35.0, 84.0, 111.0, 43.0, 33.0, 84.0, 111.0, 43.0, 34.0, 84.0, 111.0, 43.0, 35.0, 84.0, 111.0, 42.0, 32.0, 83.0, 111.0, 43.0, 32.0, 84.0, 111.0,
                       43.0, 35.0, 84.0, 111.0, 43.0, 32.0, 84.0, 111.0, 45.0, 30.0, 85.0, 111.0, 46.0, 31.0, 85.0, 111.0, 45.0, 31.0, 85.0, 111.0, 46.0, 32.0, 85.0, 111.0, 43.0, 32.0, 83.0, 111.0,
                       45.0, 31.0, 85.0, 111.0, 46.0, 33.0, 85.0, 111.0, 46.0, 33.0, 85.0, 111.0, 47.0, 25.0, 91.0, 108.0, 47.0, 25.0, 90.0, 109.0, 46.0, 25.0, 90.0, 109.0, 46.0, 27.0, 90.0, 109.0,
                       46.0, 27.0, 91.0, 108.0, 46.0, 27.0, 90.0, 108.0, 46.0, 28.0, 90.0, 110.0, 46.0, 28.0, 90.0, 109.0, 46.0, 26.0, 90.0, 98.0, 45.0, 28.0, 89.0, 99.0, 45.0, 27.0, 89.0, 98.0, 45.0,
                       27.0, 89.0, 98.0, 46.0, 27.0, 89.0, 98.0, 46.0, 27.0, 89.0, 98.0, 46.0, 28.0, 90.0, 98.0, 46.0, 28.0, 90.0, 98.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [15.0, 1.0398486852645874, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 45.0, 29.0, 85.0, 111.0, 46.0, 28.0, 85.0, 111.0, 45.0, 28.0, 85.0, 110.0, 46.0, 29.0, 85.0, 110.0, 42.0, 27.0, 82.0, 110.0, 45.0, 26.0, 85.0, 108.0, 46.0, 29.0, 85.0,
                       109.0, 46.0, 27.0, 85.0, 109.0, 46.0, 27.0, 84.0, 111.0, 47.0, 31.0, 86.0, 111.0, 46.0, 31.0, 85.0, 111.0, 46.0, 27.0, 85.0, 111.0, 46.0, 27.0, 84.0, 111.0, 45.0, 27.0, 84.0,
                       111.0, 46.0, 30.0, 84.0, 111.0, 45.0, 28.0, 84.0, 111.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0],
                      [15.0, 1.2858054637908936, 60.0, 0.0, 111.0, 111.0, 59.0, 0.0, 111.0, 111.0, 60.0, 0.0, 111.0, 111.0, 60.0, 0.0, 111.0, 111.0, 59.0, 0.0, 111.0, 111.0, 60.0, 0.0, 111.0, 111.0,
                       59.0, 0.0, 111.0, 111.0, 60.0, 0.0, 111.0, 111.0, 60.0, 16.0, 111.0, 111.0, 60.0, 15.0, 111.0, 111.0, 60.0, 15.0, 111.0, 111.0, 60.0, 15.0, 111.0, 111.0, 60.0, 15.0, 111.0,
                       111.0, 60.0, 15.0, 111.0, 111.0, 60.0, 16.0, 111.0, 111.0, 60.0, 15.0, 111.0, 111.0, 60.0, 16.0, 111.0, 111.0, 60.0, 16.0, 111.0, 111.0, 60.0, 15.0, 111.0, 111.0, 60.0, 15.0,
                       111.0, 111.0, 60.0, 15.0, 111.0, 111.0, 60.0, 15.0, 111.0, 111.0, 60.0, 16.0, 111.0, 111.0, 60.0, 16.0, 111.0, 111.0, 60.0, 16.0, 111.0, 111.0, 60.0, 16.0, 111.0, 111.0, 60.0,
                       15.0, 111.0, 111.0, 60.0, 15.0, 111.0, 111.0, 60.0, 15.0, 111.0, 111.0, 60.0, 15.0, 111.0, 111.0, 60.0, 16.0, 111.0, 111.0, 60.0, 16.0, 111.0, 111.0, 60.0, 16.0, 111.0, 111.0,
                       60.0, 16.0, 111.0, 111.0, 60.0, 15.0, 111.0, 111.0, 60.0, 15.0, 111.0, 111.0, 60.0, 15.0, 111.0, 111.0, 60.0, 15.0, 111.0, 111.0, 60.0, 16.0, 111.0, 111.0, 60.0, 16.0, 111.0,
                       111.0],
                      [15.0, 1.5366778373718262, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 40.0, 111.0,
                       111.0, 39.0, 40.0, 111.0, 111.0, 40.0, 39.0, 111.0, 111.0, 40.0, 40.0, 111.0, 111.0, 39.0, 39.0, 111.0, 111.0, 40.0, 39.0, 111.0, 111.0, 39.0, 40.0, 111.0, 111.0, 40.0, 39.0,
                       111.0, 111.0, 40.0, 40.0, 111.0, 111.0, 40.0, 40.0, 111.0, 111.0, 40.0, 39.0, 111.0, 111.0, 40.0, 40.0, 111.0, 111.0, 40.0, 39.0, 111.0, 111.0, 40.0, 39.0, 111.0, 111.0, 40.0,
                       40.0, 111.0, 111.0, 40.0, 40.0, 111.0, 111.0]])
    t2 = torch.Tensor([[18.0, 55.0, 33.0, 90.0, 98.0, 54.0, 32.0, 91.0, 98.0, 54.0, 32.0, 91.0, 98.0, 50.0, 32.0, 93.0, 98.0, 50.0, 32.0, 93.0, 98.0, 46.0, 32.0, 93.0, 98.0, 45.0, 31.0, 97.0, 98.0,
                        45.0, 32.0, 97.0, 98.0, 45.0, 31.0, 98.0, 98.0, 45.0, 30.0, 97.0, 98.0, 45.0, 30.0, 97.0, 98.0, 44.0, 30.0, 98.0, 98.0, 44.0, 30.0, 98.0, 98.0, 46.0, 30.0, 98.0, 98.0, 46.0,
                        30.0, 98.0, 98.0, 41.0, 30.0, 97.0, 98.0, 43.0, 30.0, 96.0, 98.0, 43.0, 30.0, 95.0, 98.0, 43.0, 30.0, 96.0, 98.0, 43.0, 30.0, 96.0, 98.0, 39.0, 30.0, 96.0, 98.0, 39.0, 30.0,
                        96.0, 98.0, 39.0, 30.0, 96.0, 98.0, 39.0, 30.0, 96.0, 98.0, 39.0, 30.0, 95.0, 98.0, 39.0, 30.0, 95.0, 98.0, 39.0, 30.0, 95.0, 98.0, 39.0, 30.0, 95.0, 98.0, 39.0, 30.0, 95.0,
                        98.0, 39.0, 29.0, 95.0, 98.0, 39.0, 29.0, 95.0, 98.0, 39.0, 29.0, 95.0, 98.0, 39.0, 29.0, 95.0, 98.0, 39.0, 28.0, 95.0, 98.0, 40.0, 27.0, 96.0, 98.0, 40.0, 27.0, 96.0, 98.0,
                        38.0, 27.0, 96.0, 98.0, 38.0, 28.0, 97.0, 98.0, 38.0, 30.0, 96.0, 98.0, 40.0, 30.0, 96.0, 98.0]])
    t3 = torch.Tensor([[11.0, 2.0429630279541016, 12.0, 56.0, 111.0, 111.0, 11.0, 55.0, 111.0, 111.0, 12.0, 55.0, 111.0, 111.0, 12.0, 55.0, 111.0, 111.0, 11.0, 55.0, 111.0, 111.0, 12.0, 55.0, 111.0,
                        111.0, 11.0, 56.0, 111.0, 111.0, 12.0, 55.0, 111.0, 111.0, 12.0, 56.0, 111.0, 111.0, 12.0, 56.0, 111.0, 111.0, 12.0, 55.0, 111.0, 111.0, 12.0, 55.0, 111.0, 111.0, 12.0, 55.0,
                        111.0, 111.0, 12.0, 55.0, 111.0, 111.0, 12.0, 56.0, 111.0, 111.0, 12.0, 56.0, 111.0, 111.0, 12.0, 56.0, 111.0, 111.0, 12.0, 56.0, 111.0, 111.0, 12.0, 55.0, 111.0, 111.0, 12.0,
                        55.0, 111.0, 111.0, 12.0, 55.0, 111.0, 111.0, 12.0, 55.0, 111.0, 111.0, 12.0, 56.0, 111.0, 111.0, 12.0, 56.0, 111.0, 111.0, 12.0, 56.0, 111.0, 111.0, 12.0, 56.0, 111.0, 111.0,
                        12.0, 55.0, 111.0, 111.0, 12.0, 55.0, 111.0, 111.0, 12.0, 55.0, 111.0, 111.0, 11.0, 55.0, 111.0, 111.0, 12.0, 56.0, 111.0, 111.0, 12.0, 56.0, 111.0, 111.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [11.0, 1.7257298231124878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 40.0, 40.0, 111.0, 111.0, 39.0, 40.0, 111.0, 111.0, 40.0, 39.0, 111.0, 111.0, 40.0, 40.0, 111.0, 111.0, 39.0, 39.0, 111.0, 111.0, 40.0, 39.0, 111.0, 111.0, 39.0, 40.0,
                        111.0, 111.0, 40.0, 39.0, 111.0, 111.0, 40.0, 40.0, 111.0, 111.0, 40.0, 40.0, 111.0, 111.0, 40.0, 39.0, 111.0, 111.0, 40.0, 40.0, 111.0, 111.0, 40.0, 39.0, 111.0, 111.0, 40.0,
                        39.0, 111.0, 111.0, 40.0, 40.0, 111.0, 111.0, 40.0, 40.0, 111.0, 111.0, 40.0, 40.0, 111.0, 111.0, 40.0, 40.0, 111.0, 111.0, 40.0, 39.0, 111.0, 111.0, 40.0, 40.0, 111.0, 111.0,
                        40.0, 39.0, 111.0, 111.0, 40.0, 39.0, 111.0, 111.0, 40.0, 40.0, 111.0, 111.0, 40.0, 40.0, 111.0, 111.0, 24.0, 40.0, 111.0, 111.0, 24.0, 40.0, 111.0, 111.0, 24.0, 39.0, 111.0,
                        111.0, 24.0, 40.0, 111.0, 111.0, 24.0, 39.0, 111.0, 111.0, 24.0, 39.0, 111.0, 111.0, 24.0, 40.0, 111.0, 111.0, 24.0, 40.0, 111.0, 111.0],
                       [11.0, 1.0153050422668457, 62.0, 38.0, 81.0, 83.0, 61.0, 37.0, 80.0, 82.0, 61.0, 37.0, 80.0, 82.0, 61.0, 37.0, 80.0, 82.0, 61.0, 37.0, 80.0, 82.0, 62.0, 37.0, 80.0, 82.0, 61.0,
                        37.0, 80.0, 81.0, 62.0, 38.0, 80.0, 81.0, 61.0, 37.0, 81.0, 82.0, 61.0, 39.0, 81.0, 81.0, 61.0, 38.0, 81.0, 81.0, 61.0, 38.0, 81.0, 82.0, 62.0, 38.0, 81.0, 82.0, 62.0, 38.0,
                        81.0, 81.0, 62.0, 39.0, 81.0, 81.0, 62.0, 39.0, 81.0, 81.0, 62.0, 37.0, 81.0, 95.0, 62.0, 39.0, 81.0, 96.0, 62.0, 40.0, 81.0, 94.0, 62.0, 38.0, 81.0, 93.0, 62.0, 37.0, 81.0,
                        93.0, 62.0, 36.0, 81.0, 94.0, 62.0, 34.0, 83.0, 88.0, 62.0, 32.0, 84.0, 88.0, 60.0, 27.0, 88.0, 77.0, 62.0, 29.0, 88.0, 72.0, 65.0, 27.0, 88.0, 67.0, 65.0, 29.0, 87.0, 65.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [4.0, 1.1674189567565918, 62.0, 45.0, 82.0, 96.0, 61.0, 45.0, 83.0, 96.0, 61.0, 44.0, 83.0, 96.0, 61.0, 42.0, 83.0, 94.0, 61.0, 42.0, 82.0, 94.0, 62.0, 44.0, 82.0, 96.0, 61.0,
                        42.0, 82.0, 93.0, 62.0, 42.0, 83.0, 93.0, 62.0, 42.0, 82.0, 93.0, 62.0, 42.0, 83.0, 93.0, 62.0, 42.0, 82.0, 93.0, 62.0, 42.0, 82.0, 93.0, 62.0, 43.0, 82.0, 93.0, 61.0, 42.0,
                        82.0, 93.0, 62.0, 42.0, 83.0, 93.0, 62.0, 42.0, 82.0, 94.0, 62.0, 32.0, 82.0, 81.0, 62.0, 32.0, 82.0, 81.0, 62.0, 32.0, 83.0, 81.0, 62.0, 32.0, 82.0, 81.0, 62.0, 32.0, 83.0,
                        82.0, 62.0, 32.0, 82.0, 80.0, 62.0, 33.0, 82.0, 81.0, 62.0, 32.0, 82.0, 81.0, 63.0, 34.0, 83.0, 80.0, 63.0, 34.0, 83.0, 81.0, 63.0, 34.0, 83.0, 80.0, 64.0, 35.0, 82.0, 80.0,
                        63.0, 35.0, 83.0, 80.0, 63.0, 33.0, 82.0, 80.0, 63.0, 36.0, 82.0, 80.0, 63.0, 36.0, 82.0, 80.0, 63.0, 36.0, 82.0, 76.0, 63.0, 36.0, 82.0, 76.0, 63.0, 36.0, 82.0, 76.0, 63.0,
                        36.0, 82.0, 76.0, 63.0, 36.0, 82.0, 76.0, 63.0, 37.0, 82.0, 76.0, 64.0, 38.0, 82.0, 76.0, 64.0, 37.0, 82.0, 76.0]])
    t4= torch.Tensor([[4.0, 58.0, 51.0, 82.0, 88.0, 56.0, 51.0, 83.0, 87.0, 55.0, 51.0, 83.0, 87.0, 56.0, 51.0, 81.0, 87.0, 57.0, 51.0, 81.0, 86.0, 57.0, 51.0, 80.0, 85.0, 58.0, 48.0, 82.0, 84.0,
                       58.0, 47.0, 84.0, 84.0, 58.0, 45.0, 85.0, 84.0, 62.0, 45.0, 85.0, 83.0, 67.0, 45.0, 82.0, 85.0, 59.0, 45.0, 82.0, 84.0, 58.0, 45.0, 86.0, 82.0, 57.0, 40.0, 88.0, 83.0, 61.0,
                       37.0, 85.0, 84.0, 65.0, 36.0, 84.0, 80.0, 62.0, 30.0, 86.0, 82.0, 62.0, 26.0, 89.0, 81.0, 65.0, 28.0, 90.0, 80.0, 65.0, 28.0, 93.0, 80.0, 64.0, 28.0, 94.0, 80.0, 64.0, 29.0,
                       92.0, 78.0, 65.0, 32.0, 90.0, 79.0, 65.0, 36.0, 88.0, 78.0, 65.0, 38.0, 87.0, 78.0, 66.0, 38.0, 87.0, 77.0, 63.0, 37.0, 86.0, 78.0, 61.0, 38.0, 88.0, 76.0, 63.0, 36.0, 91.0,
                       76.0, 64.0, 36.0, 91.0, 74.0, 67.0, 34.0, 93.0, 70.0, 67.0, 35.0, 93.0, 66.0, 64.0, 35.0, 93.0, 62.0, 65.0, 35.0, 91.0, 63.0, 67.0, 35.0, 89.0, 65.0, 67.0, 35.0, 89.0, 70.0,
                       67.0, 35.0, 89.0, 72.0, 67.0, 35.0, 89.0, 71.0, 67.0, 35.0, 89.0, 71.0, 67.0, 35.0, 89.0, 69.0]])
    groundtruth_dic = {}
    detection_dic = {}
    groundtruth_dic['FIFA'] = t4
    detection_dic['FIFA'] = t3
    groundtruth_dic['mam'] = t2
    detection_dic['mam'] = t

    # print('t.shape :',t.shape)
    # print('t2.shape :',t2.shape)
    # print('t3.shape :',t3.shape)
    # print('t4.shape :',t4.shape)
    
    calculate_mAP(detection_dic, groundtruth_dic, 0.5)


    
