# -*- coding:utf-8 -*-

###updated in gitee

import random   #
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
import xlrd
import xlwt
import math

#1--indicate all the single slips have saved in local, and will skip the corresponding steps and will save time; 
#0--indicate all the single slips have not saved yet , the algorithm will separate every slips from a whole page image
slips_already_saved = 0   

#1--show the multi-Gaussian fit result of every character for 3 seconds;  
#0--not show the figure                         
is_show_figure = 0        
                          
#1 Guassian model
def Guassian1(x, a1, m1, s1):
    return a1 * np.exp(-((x - m1) / s1) ** 2)

#3 Guassian model
def Guassian3(x, a1, a2, a3, m1, m2, m3, s1, s2, s3):
    return a1 * np.exp(-((x - m1) / s1) ** 2) + a2 * np.exp(-((x - m2) / s2) ** 2) + a3 * np.exp(-((x - m3) / s3) ** 2)

print("-------------Start to segment the bamboo slips-------------")

image = cv2.imread("./data/Image1.png")
height,width = image.shape[:2]
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

out_img_color = np.zeros((height, width),dtype=np.uint8)
out_img_color = cv2.cvtColor(out_img_color, cv2.COLOR_GRAY2BGR)
recoverd_img = np.zeros((height, width),dtype=np.uint8)
whole_character_pos = [[] for i in range(2)]
print("start process......")
number_of_slips = 0

if slips_already_saved == 0:
    img_gray0 = []
    img_gray1 = []
    for i in range(height):
        for j in range(width):
            if image_gray[i,j] < 200:
                img_gray0.append(255)
            else:
                img_gray0.append(0)

    img_gray = np.asarray(img_gray0, dtype=np.uint8).reshape(height,width) #如果不加这个dtype=np.uint8 会出错

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    eroded = cv2.erode(img_gray,kernel)
    dilated = cv2.dilate(eroded,kernel)
    dilated2 = cv2.dilate(dilated,kernel)
    contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    #Extract every single slip ,straighten it and align left
    cnt = 0
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if w*h > 15000:
            out_img = np.zeros((height, width), dtype=np.uint8)
            out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(out_img, contours, i, (255, 255, 255), cv2.FILLED)

            out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)

            max_width = 0          #the max width of current slip
            total_valid_line = 0   
            first_line_flag = 0    
            first_line = 0         
            for i in range(height):
                k = 0
                first_flag = 0
                for j in range(width):
                    if out_img[i,j] > 200:
                        if first_line_flag == 0:  
                            first_line_flag = 1
                            first_line = i        
                        if first_flag == 0:       
                            first_flag = 1
                            total_valid_line = total_valid_line + 1
                            k = j
                        out_img_color[i, j-k, 0] = image[i, j, 0]  #Align left
                        out_img_color[i, j-k, 1] = image[i, j, 1]  #Align left
                        out_img_color[i, j-k, 2] = image[i, j, 2]  #Align left
                        if (j-k) > max_width:  
                            max_width = j-k
            
            roi = out_img_color[first_line:first_line+total_valid_line,0:max_width]
            
            roi_file_name = './result/roi/roi' + str(cnt) + '.jpg'
            cv2.imwrite(roi_file_name, roi)
            print("roi",cnt,"saved")
            whole_character_pos[0].append(x)
            whole_character_pos[1].append(y)
            cnt = cnt + 1

    #whole_character_pos saves the left top position of the every slip
    print("whole_list is:",whole_character_pos)
    number_of_slips = len(whole_character_pos[0])

    print("There are", number_of_slips,"Slips!")

    para_data = xlwt.Workbook(encoding = 'utf-8')
    para_data_sheet = para_data.add_sheet('parameters')
    para_data_sheet.write(0,0, label = 'x')
    para_data_sheet.write(0,1, label = 'y')
    for para_cnt in range(number_of_slips):
        para_data_sheet.write(para_cnt + 1, 0, whole_character_pos[0][para_cnt])
        para_data_sheet.write(para_cnt + 1, 1, whole_character_pos[1][para_cnt])
    para_data.save('./result/para.xls')

else:
    para_data_read = xlrd.open_workbook("./result/para.xls")
    para_sheet = para_data_read.sheet_by_name("parameters")

    print("para_sheet.nrows = ", para_sheet.nrows)
    print("para_sheet.ncols = ", para_sheet.ncols)
    print("para_sheet.cell_value(1, 2) = ", para_sheet.cell_value(1, 0))
    for para_cnt in range(para_sheet.nrows-1):
        whole_character_pos[0].append(int(para_sheet.cell_value(para_cnt+1, 0)))
        whole_character_pos[1].append(int(para_sheet.cell_value(para_cnt+1, 1)))
    number_of_slips = len(whole_character_pos[0])


total_cnt = 0
for total_cnt in range(number_of_slips):
    roi_file_name = './result/roi/roi' + str(total_cnt) + '.jpg'
    roi = cv2.imread(roi_file_name)
    height_roi,width_roi = roi.shape[:2]
    roi_color = roi.copy()
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    final_output = np.zeros((height_roi, width_roi),dtype=np.uint8)


    img_gray_roi0 = []

    threshold = 130
    threshold2 = 130

    (mean, stddv) = cv2.meanStdDev(roi)
    mean = mean*0.8
    print("mean=",mean)
    print("stddv=",stddv)

    for i in range(height_roi):
        for j in range(width_roi):
            temp_threshold = mean
            if 10 < roi[i,j] < temp_threshold:
                img_gray_roi0.append(255)
            else:
                img_gray_roi0.append(0)
    img_gray_roi = np.asarray(img_gray_roi0, dtype=np.uint8).reshape(height_roi,width_roi)

    img_hist_roi = np.zeros((height_roi, width_roi),dtype=np.uint8)

    
    #Horizontal histogram projection
    for i in range(height_roi):
        hist_cnt = 0
        for j in range(width_roi):
            if img_gray_roi[i,j] > 200:  # 10 < roi[i,j] < threshold
                hist_cnt = hist_cnt + 1
        for k in range(hist_cnt):
            img_hist_roi[i,k] = 255

    mask_length = 24
    mask_line_pos = 5

    img_hist_roi_color = img_hist_roi.copy()
    img_hist_roi_color = cv2.cvtColor(img_hist_roi_color, cv2.COLOR_GRAY2BGR)
    cv2.line(img_hist_roi_color,(mask_line_pos,0),(mask_line_pos,height_roi),(255,0,0),1)

	#Segment every character according to the horizontal histogram 
    rise_cnt = 0
    rise_pos = 0
    fall_cnt = 0
    fall_pos = 0
    total_edge_cnt = 1
    edge_data = []
    for i in range(height_roi):
        j = mask_line_pos
        if height_roi-mask_length/2 > i > mask_length/2:
            raise_edge_n = 0
            raise_edge_p = 0
            rall_edge_n = 0
            rall_edge_p = 0
            sum1 = 0
            sum2 = 0
            
            for k in range(int(mask_length/2)):
                if img_hist_roi[i-k,j] < 100:
                    raise_edge_n = raise_edge_n + 1
                    sum1 = sum1 + img_hist_roi[i-k,j]
                else:
                    rall_edge_p = rall_edge_p + 1
                    sum1 = sum1 + img_hist_roi[i - k, j]
                if img_hist_roi[i+k,j] > 100:
                    raise_edge_p = raise_edge_p + 1
                    sum2 = sum2 + img_hist_roi[i + k, j]
                else:
                    rall_edge_n = rall_edge_n + 1
                    sum2 = sum2 + img_hist_roi[i + k, j]
            
            if abs(raise_edge_n - raise_edge_p) < 2 and sum1 < sum2:
                if fall_cnt > 0:
                    fall_pos = int(fall_pos/fall_cnt)
                    edge_data.append(fall_pos)
                    cv2.line(img_hist_roi_color, (0, fall_pos), (width_roi, fall_pos), (0, 255, 0), 1)
                    cv2.line(roi_color, (0, fall_pos), (width_roi, fall_pos), (0, 255, 0), 1)
                    
                    total_edge_cnt = total_edge_cnt + 1
                    fall_cnt = 0
                    fall_pos = 0

                rise_cnt = rise_cnt + 1
                rise_pos = rise_pos + i
            
            if abs(rall_edge_n - rall_edge_p) < 2 and sum1 > sum2:
                if rise_cnt > 0:
                    rise_pos = int(rise_pos / rise_cnt)
                    edge_data.append(rise_pos)
                    cv2.line(img_hist_roi_color, (0, rise_pos), (width_roi, rise_pos), (0, 0, 255), 1)
                    cv2.line(roi_color, (0, rise_pos), (width_roi, rise_pos), (0, 0, 255), 1)
                    
                    total_edge_cnt = total_edge_cnt + 1
                    rise_cnt = 0
                    rise_pos = 0
                fall_cnt = fall_cnt + 1
                fall_pos = fall_pos + i
    
    if rise_cnt > 0:
        rise_pos = int(rise_pos / rise_cnt)
        rise_pos = rise_pos + 20
        if rise_pos > height_roi:
            rise_pos = height_roi
        edge_data.append(rise_pos)
        cv2.line(img_hist_roi_color, (0, rise_pos), (width_roi, rise_pos), (0, 0, 255), 1)
        cv2.line(roi_color, (0, rise_pos), (width_roi, rise_pos), (0, 0, 255), 1)
        
        total_edge_cnt = total_edge_cnt + 1
        rise_cnt = 0
        rise_pos = 0
    
    if fall_cnt > 0:
        fall_pos = int(fall_pos / fall_cnt)
        fall_pos = fall_pos + 20
        if fall_pos > height_roi:
            fall_pos = height_roi
        edge_data.append(fall_pos)
        cv2.line(img_hist_roi_color, (0, fall_pos), (width_roi, fall_pos), (0, 255, 0), 1)
        cv2.line(roi_color, (0, fall_pos), (width_roi, fall_pos), (0, 255, 0), 1)
        
        total_edge_cnt = total_edge_cnt + 1
        fall_cnt = 0
        fall_pos = 0

    
    if len(edge_data)<3:
        print("This may be a blank slip!!!")
    else:
        print("-------------------Begin to process the", total_cnt + 1, "th slip-------------------")
        splite_data = []
        splite_data.append(edge_data[0])
        for i in range(1,len(edge_data)-2,2):
            pos = int((edge_data[i] + edge_data[i+1])/2)
            splite_data.append(pos)
        splite_data.append(edge_data[-1])
        for i in range(len(splite_data)):
            cv2.line(roi_color, (0, splite_data[i]), (width_roi, splite_data[i]), (255, 255, 255), 1)

        
        print("There are ", len(splite_data)-1,"characters in total")
        max_mean_output = []
        for k in range(0,len(splite_data)-1):
            plot_data_x = []
            plot_data_y = []

            roi_new = roi[splite_data[k]:splite_data[k+1],0:width_roi]
            #cv2.imwrite("./result/roi_new.jpg",roi_new)
            height_roi_new,width_roi_new = roi_new.shape[:2]
            roi_new_copy = np.zeros((height_roi_new, width_roi_new),dtype=np.uint8)

            roi_new_copy = roi_new.copy()
            (mean, stddv) = cv2.meanStdDev(roi_new_copy)

            no_character_flag = 0
            mean_int = mean[0]
            stddv_int = int(stddv[0])
            
            if stddv_int < 22:
                no_character_flag = 1
            
            start_thres = 0.6*int(mean_int)
            start_thres = int(start_thres)
            print("start:",start_thres)
            stop_thre = 1.2*int(mean_int)
            if stop_thre > 255:
                stop_thre = 255
            stop_thre = int(stop_thre)
            print("stop:",stop_thre)

            data_index = 1
            max_mean = 0
            max_mean_threshold = 0

            for thres in range(start_thres, stop_thre):
                cnt = 0
                roi_new_binary = np.zeros((height_roi_new, width_roi_new), dtype=np.uint8)
                for i in range(height_roi_new):
                    for j in range(width_roi_new):
                        if 10 < roi_new_copy[i,j] < thres:
                            cnt = cnt + 1
                            roi_new_binary[i,j] = 255

                contours, hier = cv2.findContours(roi_new_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1) #RETR_EXTERNAL
                size_total = 0
                size_mean = 0
                count = 0
                count_small = 0
                count_large = 0
                max_length = 0
                for i in range(len(contours)):
                    #calculate the contour length
                    contour_length = cv2.arcLength(contours[i], True)
                    #area = cv2.contourArea(contours[i])
                    if contour_length > max_length:
                        max_length = contour_length
                    #count the number of small contours and large contours 
                    if contour_length < 10:
                        count_small = count_small + 1
                    else:
                        count_large = count_large + 1
                        size_total = size_total + cv2.arcLength(contours[i], True)
                    count = count + 1
                #calculateECCL value(Effective Character Contour Length)
                if count > 0:
                    size_mean = size_total/count
                    if size_mean > max_mean:
                        max_mean = size_mean
                        max_mean_threshold = thres
                
                plot_data_x.append(thres)
                plot_data_y.append(size_mean)

                data_index = data_index + 1

            if no_character_flag == 0:
                #############################################################
                #### begin multi-Guassian fitting of ECCL values
                #############################################################
                x_np_data = np.array(plot_data_x)
                y_np_data = np.array(plot_data_y)
                max_y = max(plot_data_y)
                
                popt, pcov = curve_fit(Guassian3, x_np_data, y_np_data,
                                       bounds=([0, 0, 0, 0, 0, 0, 0.01, 0.01, 0.01],
                                               [2*max_y, 2*max_y, 2*max_y, 400, 400, 400, 200, 200, 200]),maxfev=5000)
                a1 = popt[0]
                a2 = popt[1]
                a3 = popt[2]
                u1 = popt[3]
                u2 = popt[4]
                u3 = popt[5]
                sig1 = popt[6]
                sig2 = popt[7]
                sig3 = popt[8]
                print('a1=', a1, 'a2=', a2, 'a3=', a3)
                print('u1=', u1, 'u2=', u2, 'u3=', u3)
                print('sig1=', sig1, 'sig2=', sig2, 'sig3=', sig3)
                # yvals = Guassian3(x_np_data, a1, u1, sig1, a2, u2, sig2, a3, u3, sig3)  # 拟合y值
                yvals = Guassian3(x_np_data, a1, a2, a3, u1, u2, u3, sig1, sig2, sig3)

                
                yvals_1 = Guassian1(x_np_data, a1, u1, sig1)
                yvals_2 = Guassian1(x_np_data, a2, u2, sig2)
                yvals_3 = Guassian1(x_np_data, a3, u3, sig3)

                # find the peaks of the multi-Gussian curve
                num_peak = signal.find_peaks(yvals, distance=10)

                plt.plot(plot_data_x, plot_data_y, 'b', label='Original')
                plt.plot(plot_data_x, yvals, 'r', label='Multi Guassian')

                
                plt.plot(plot_data_x, yvals_1, 'y', linestyle=':', label='Guassian1')
                plt.plot(plot_data_x, yvals_2, 'c', linestyle=':', label='Guassian2')
                plt.plot(plot_data_x, yvals_3, 'm', linestyle=':', label='Guassian3')

                max_peak = 0
                max_peak_pos = 0
                for ii in range(len(num_peak[0])):
                    plt.plot(num_peak[0][ii] + plot_data_x[0], yvals[num_peak[0][ii]], '*', markersize=10)
                    if (yvals[num_peak[0][ii]]) > max_peak:
                        max_peak_pos = num_peak[0][ii] + plot_data_x[0]
                        max_peak = yvals[num_peak[0][ii]]
                    # print('fitted peak pos=',num_peak[0][ii]+plot_data_x[0],'peak=',yvals[num_peak[0][ii]])

                if is_show_figure == 1:
                    plt.legend()
                    plt.xlabel('Threshold')
                    plt.ylabel('Effective Contour Length')
                    plt.title('Multi Gaussian Fitting')  # Gaussian Fitting
                    plt.savefig('./result/GaussianFit.jpg')
                    plt.pause(3)
                    plt.close()

                print("##########Before fitted max_mean_theshold=", max_mean_threshold, '##########')
                max_mean_threshold = max_peak_pos  
                #############################################################
                #### end of multi-Guassian fitting of ECCL values
                #############################################################

                max_mean_output.append(max_mean_threshold)
                print("##########After fitted max_mean_theshold=", max_mean_threshold, '##########')
                print("max_mean=", max_mean)

                valid_point_cnt = 0
                for i in range(height_roi_new):
                    for j in range(width_roi_new):
                        if 10 < roi_new_copy[i, j] < max_mean_threshold and no_character_flag == 0:
                            final_output[splite_data[k] + i, j] = 255
                            recoverd_img[whole_character_pos[1][total_cnt] + splite_data[k] + i, whole_character_pos[0][total_cnt] + j] = 255
                            valid_point_cnt = valid_point_cnt + 1
                
                if valid_point_cnt > 3 * height_roi_new * width_roi_new / 4:
                    print("#############This region may be mistake, reset to background again###############")
                    for i in range(height_roi_new):
                        for j in range(width_roi_new):
                            if 10 < roi_new_copy[i, j] < max_mean_threshold and no_character_flag == 0:
                                final_output[splite_data[k] + i, j] = 0
                                recoverd_img[whole_character_pos[1][total_cnt] + splite_data[k] + i, whole_character_pos[0][total_cnt] + j] = 0
                print("The", k + 1, "th character segmented done!")
            else:
                print("The", k + 1, "th character have been skipped")
        print("-------------------The", total_cnt + 1, "th character segmented and restored done!-------------------")

print("Process done......")
#cv2.imwrite("./result/img_gray_roi.jpg",img_gray_roi)
#cv2.imwrite("./result/img_hist.jpg",img_hist_roi_color)
#cv2.imwrite("./result/roi_color.jpg",roi_color)
#cv2.imwrite("./result/final_output.jpg",final_output)
cv2.imwrite("./result/recoverd_img.jpg",recoverd_img)


