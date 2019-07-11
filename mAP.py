import glob
import json
import os
import shutil
import operator
import sys
import argparse
import math
import torch

import numpy as np

min_overlap = 0.3
iou_overlap = 0.5
iou_overlap_4 = 0.4
iou_overlap_3 = 0.3

"""
 throw error and exit
"""
def error(msg):
    print(msg)
    sys.exit(0)

def is_float_between_0_and_1(value):
    try:
        val = float(value)
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


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

"""
 Convert the lines of a file to a list
"""
def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

"""
 Convert the lines of a file to a list
"""
def file_json_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = json.load(f)
    return content


if __name__ == '__main__':


    GT_PATH = os.path.join(os.getcwd(), 'outputs', 'groundtruth')
    DR_PATH = os.path.join(os.getcwd(), 'outputs', 'detection')

    # get a list with the ground-truth files
    ground_truth_files_list = glob.glob(GT_PATH + '/*.json')

    gt_counter_per_class = {}
    counter_videos_per_class = {}
    gt_files = {}


    for json_file in ground_truth_files_list:
        #print(txt_file)

        file_id = json_file.split(".json", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))

        # check if there is a correspondent detection-results file
        gt_tubes_list = file_json_to_list(json_file)

        # create ground-truth dictionary
        gt_bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for tub in gt_tubes_list:
            class_id = tub[0]
            tube = torch.Tensor(tub[1:]).view(-1,4)
            gt_bounding_boxes.append({"class_id":class_id, "tube":tube, "used":False})
            # count that object
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

        gt_files[file_id] = gt_bounding_boxes

    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    detected_bboxes = {}

    # get a list with the detection-results files
    dr_files_list = glob.glob(DR_PATH + '/*.json')

    dr_files_list.sort()
    print('detection...')
    for class_index, class_id in enumerate(gt_classes):
        bounding_boxes = []
        for json_file in dr_files_list:
            #print(txt_file)
            
            # the first time it checks if all the corresponding ground-truth files exist
            file_id = json_file.split(".json",1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            print('file_id :',file_id)
            dt_tubes_list = file_json_to_list(json_file)

            for tub in dt_tubes_list:

                tmp_class_id = tub[0]
                confidence = tub[1]
                tube = torch.Tensor(tub[2:]).view(-1,4)

                if tmp_class_id == class_id:
                    #print("match")
                    bounding_boxes.append({"confidence":confidence, "file_id":file_id, "tube":tube})

        # sort detection-results by decreasing confidence
        bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)

        detected_bboxes[class_id] = bounding_boxes

    # for i in detected_bboxes.keys():
    #     print('class_name :',i, ' bounding_boxes :',detected_bboxes[i])


    """
     Calculate the AP for each class
    """
    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    # open file to store the results
    with open( "./results.txt", 'w') as results_file:
        results_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        
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
            print('nd :',nd)
            for idx, detection in enumerate(dr_data):
                file_id = detection["file_id"]

                # assign detection-results to ground truth object if any
                # open ground-truth with that file_id
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
                        print('bi.shape :',bi.shape)
                        iw = (bi[:,2] - bi[:,0] + 1).clamp_(min=0)
                        ih = (bi[:,3] - bi[:,1] + 1).clamp_(min=0)
                        # print('iw.shape :',iw.shape)
                        # print('ih.shape :',ih.shape)
                        # print('iw :',iw)
                        # print('ih :',ih)
                        # print('dt_tube[:,0] :',dt_tube[:,0])
                        # print('gt_tube[:,0] :',gt_tube[:,0])
                        # print('dt_tube[:,2] :',dt_tube[:,2])
                        # print('gt_tube[:,2] :',gt_tube[:,2])

                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (dt_tube[:,2] - dt_tube[:,0] + 1) * (dt_tube[:,3] - dt_tube[:,1] + 1) + (gt_tube[:,2] - gt_tube[:,0]
                                                        + 1) * (gt_tube[:,3] - gt_tube[:,1] + 1) - iw * ih
                        # print('dt_tube :',dt_tube)
                        # print('gt_tube :',gt_tube)
                        # print('dt_tube.ne(0).any(dim=1) :',dt_tube.ne(0).any(dim=1).nonzero())
                        # print('gt_tube[[12,13,14,15]] :',gt_tube[[12,13,14,15]])
                        # print('dt_tube[[12,13,14,15]] :',dt_tube[[12,13,14,15]])

                        ov = iw * ih / ua
                        ov.masked_fill_(zero_area,0)

                        ov = ov.sum()/ (dt_tube.size(0) - empty_indices.nelement())
                        print('ov :',ov)
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj


                # set minimum overlap
                min_overlap = min_overlap

                if ovmax >= min_overlap:
                    print('gt_match :',gt_match)

                    if not bool(gt_match["used"]):
                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_id] += 1
                        # # update the ".json" file
                        # with open(gt_file, 'w') as f:
                        #     f.write(json.dumps(ground_truth_data))
                    else:
                        # false positive (multiple detection)
                        fp[idx] = 1

                else:
                    # false positive
                    fp[idx] = 1
                    if ovmax > 0:
                        status = "INSUFFICIENT OVERLAP"

            #print(tp)
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
            """
             Write to results.txt
            """
            rounded_prec = [ '%.2f' % elem for elem in prec ]
            rounded_rec = [ '%.2f' % elem for elem in rec ]
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
            print(text)
            ap_dictionary[class_id] = ap

            n_videos = counter_videos_per_class[class_id]
            lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_videos)
            lamr_dictionary[class_id] = lamr


        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes
        text = "mAP = {0:.2f}%".format(mAP*100)
        results_file.write(text + "\n")
        print(text)

    # iterate through all the files
    print('dr_files_list :',dr_files_list)
    det_counter_per_class = {}
    for txt_file in dr_files_list:
        # get lines to list
        lines_list = file_json_to_list(txt_file)
        for idx, line in enumerate(lines_list):
            print('idx :', idx, ' line :',line[0])
            class_id = line[0]
            # count that object
            if class_id in det_counter_per_class:
                det_counter_per_class[class_id] += 1
            else:
                # if class didn't exist yet
                det_counter_per_class[class_id] = 1
    print('count_true_positives :',count_true_positives)
    print('det_counter_per_class :',det_counter_per_class)
    #print(det_counter_per_class)
    dr_classes = list(det_counter_per_class.keys())

    with open( "./results.txt", 'a') as results_file:
        results_file.write("\n# Number of ground-truth objects per class\n")
        for class_name in sorted(gt_counter_per_class):
            results_file.write(str(class_name) + ": " + str(gt_counter_per_class[class_name]) + "\n")

    print('dr_classes :',dr_classes)
    with open( "./results.txt", 'a') as results_file:
        results_file.write("\n# Number of detected objects per class\n")
        for class_name in sorted(dr_classes):
            if class_name == 0:
                continue
            n_det = det_counter_per_class[class_name]
            text = str(class_name) + ": " + str(n_det)
            text += " (tp:" + str(count_true_positives[class_name]) + ""
            text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
            results_file.write(text)
