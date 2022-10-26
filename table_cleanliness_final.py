from ast import arguments
from asyncio.constants import SSL_HANDSHAKE_TIMEOUT
import enum
from gc import collect

from traitlets import default
import cv2
import os
import numpy as np
import collections
import sys
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from datetime import datetime, timezone, timedelta

class Table:
    def __init__(self, id, bbox, status, timestamp):
        self.id = id
        self.bbox = bbox
        self.status = 'Clean' 
        self.last_overlapped_ts = timestamp
        self.counter = 0

def get_rectangle_start_end(pred, shape):

    x_start = int(pred.bounding_box.left * shape[1])
    y_start = int(pred.bounding_box.top * shape[0])
    x_end = x_start + int(pred.bounding_box.width * shape[1])
    y_end = y_start + int(pred.bounding_box.height * shape[0])

    return x_start, y_start, x_end, y_end


def initiate_table(results, prob, shape):
    table_list = []
    table_preds = [prediction for prediction in results.predictions if prediction.probability > prob and prediction.tag_name == 'table']

    table_preds_rect = [get_rectangle_start_end(pred, shape) for pred in table_preds]
    table_preds_rect = sorted(table_preds_rect)

    for id, table_pred in enumerate(table_preds_rect):
        table_list.append(Table(id, table_pred, 'Clean', False))

    return table_list

# You don't need to update the table bbox if your camera is fixed. I created this function because my camera kept moving.
def update_table_bbox(results, prob, shape, table_list):
    tb_preds = [prediction for prediction in results.predictions if prediction.probability > prob and prediction.tag_name == 'table']
    if len(tb_preds) < 3:
        tb_preds = sorted([prediction for prediction in results.predictions if prediction.tag_name == 'table'], key = lambda p: p.probability, reverse= True)[:3]

    tb_preds_rect = [get_rectangle_start_end(pred, shape) for pred in tb_preds]
    tb_preds_rect = sorted(tb_preds_rect)

    for i, Table in enumerate(table_list):
        table_list[i].bbox = tb_preds_rect[i]

def overlapped(tb_bbox, p_bbox):
    x_tb_start, y_tb_start, x_tb_end, y_tb_end = tb_bbox
    start_point, end_point = p_bbox
    x_p_start, y_p_start = start_point
    x_p_end, y_p_end = end_point 
    if max(x_tb_start, x_p_start) < min(x_tb_end, x_p_end) and max(y_tb_start, y_p_start) < min(y_tb_end, y_p_end):
        return True    
    return False

def overlapped_area_cal(tb_bbox, p_bbox):
    x_tb_start, y_tb_start, x_tb_end, y_tb_end = tb_bbox
    start_point, end_point = p_bbox
    x_p_start, y_p_start = start_point
    x_p_end, y_p_end = end_point  
    
    area = (min(x_tb_end, x_p_end) - max(x_tb_start, x_p_start)) * (min(y_tb_end, y_p_end) * max(y_tb_start, y_p_start))

    return area

def crop_image(x_start, y_start, x_end, y_end, img, i, table_id):
    cropped_img = img[y_start:y_end, x_start:x_end]
    cv2.imwrite(f"cropped_folder/cropped{i:04d}_{table_id:02d}.jpg", cropped_img)


def detect_table_cleanliness(i, table_id):
    # custom vision credentials information
    prediction_key = '<Your Prediction Key>'
    ENDPOINT = '<Your Endpoint>'
    credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    predictor = CustomVisionPredictionClient(endpoint=ENDPOINT, credentials=credentials)
    project_id = '<Your Project Id>'
    PUBLISH_ITERATION_NAME = '<Internation name>'

    prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)
    clean_prob = 0
    non_clean_prob = 0
    with open(os.path.join('cropped_folder', f'cropped{i:04d}_{table_id:02d}.jpg'), mode="rb") as image_contents:
            results = predictor.classify_image(
            project_id, PUBLISH_ITERATION_NAME, image_contents.read())

    for prediction in results.predictions:
        if prediction.tag_name == 'clean':
            clean_prob = prediction.probability
        else:
            non_clean_prob = prediction.probability

    if non_clean_prob >= clean_prob:
        return False
    
    else:
        return True

def update_tb_color_by_status(img, overlay, table_list):
    table_status = collections.defaultdict()
    table_status['Clean'] = (0, 204, 0)
    table_status['Occupied'] = (102, 178, 255)
    table_status['Need_clean'] = (0, 0, 255)

    for id, Table in enumerate(table_list):
        cv2.rectangle(img, (Table.bbox[0], Table.bbox[1]), (Table.bbox[2], Table.bbox[3]), table_status[Table.status], -1)  # A filled rectangle
        alpha = 0.6 
        img = cv2.addWeighted(img, alpha, overlay, 1 - alpha, 0) 

    return img

def main():
    # arguments
    argument_list = sys.argv
    print(argument_list)
    if len(argument_list) < 2:
        print('You did not assgin the probabily in the command argument. Set the probabilty to default value 50')
        prob_threshold = 0.5   
    else:
        prob_threshold = argument_list[1]
    
    # custom vision credentials information
    prediction_key = '<Your Prediction Key>'
    ENDPOINT = '<Your Endpoint>'
    credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    predictor = CustomVisionPredictionClient(endpoint=ENDPOINT, credentials=credentials)
    project_id = '<Your Project Id>'
    PUBLISH_ITERATION_NAME = '<Internation name>'

    # video path information
    video_path = './'
    extract_img_folder = '_extracted_images_m'
    video_name = '<Your Video Name>'
    parent_path = './'
    ol_t_threshold = 90
    lv_t_threshold = 625
    frame_count = extract_frames_from_video(video_path, video_name, parent_path, extract_img_folder)
    ai_inference(frame_count, extract_img_folder, predictor, project_id, PUBLISH_ITERATION_NAME, prob_threshold, parent_path, ol_t_threshold, lv_t_threshold)
    compose_video(frame_count, PUBLISH_ITERATION_NAME, prob_threshold, project_id)

def extract_frames_from_video(video_path, video_name, parent_path, extract_img_folder):
    vidcap = cv2.VideoCapture(video_path + video_name)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success,image = vidcap.read()
    if success == 'False':
        print('1. Read the video FAILED. Check if your video file exists')
    else:
        print(f'1. Read the video SUCESSFULLY. The fps of the video is {fps}')
  
    img_path = parent_path + extract_img_folder
    os.mkdir(img_path)
    frame_count = 0
    
    while success:
        cv2.imwrite(os.path.join(img_path , f'frame_{frame_count}.jpg'), image)     
        success,image = vidcap.read()
        frame_count += 1
    
    print('2. Finish extracting the video to frames')
    return frame_count

def ai_inference(frame_count, extract_img_folder, predictor, project_id, PUBLISH_ITERATION_NAME, prob_threshold, parent_path, ol_t_threshold, lv_t_threshold):

    # Open the sample image and get back the prediction results.
    # Initiate table class
    with open(os.path.join(extract_img_folder, "frame_0.jpg"), mode="rb") as test_data:
        results = predictor.detect_image(project_id, PUBLISH_ITERATION_NAME, test_data)
        print(results)
        print('3. Call the Custom Vision SUCESSFULLY')
        # initiate tables 
        img = cv2.imread(f'{extract_img_folder}/frame_0.jpg')
        shape = img.shape
        table_list = initiate_table(results, prob_threshold, shape)
  
    prob = float(prob_threshold)
    
    # created tagged folder and cropped folder
    cropped_folder = 'cropped_folder'
    path = os.path.join(parent_path, cropped_folder)
    os.mkdir(path)

    tagged_folder = f'{project_id}_tagged_images'
    path = os.path.join(parent_path, tagged_folder)
    os.mkdir(path)
 
    for i in range(frame_count):
        occupied = set() 
        with open(os.path.join(extract_img_folder, f'frame_{i}.jpg'), mode="rb") as test_data:
            results = predictor.detect_image(project_id, PUBLISH_ITERATION_NAME, test_data)
            img = cv2.imread(f'{extract_img_folder}/frame_{i}.jpg')
            overlay = img.copy()
            shape = img.shape          
            update_table_bbox(results, prob, shape, table_list) 
            now = datetime.now()
            # get people prediction results
            filtered_preds_people = [prediction for prediction in results.predictions if prediction.probability > prob and prediction.tag_name == 'people'] 
            
            # set every table status to Clean first
            for id, table in enumerate(table_list):
                table.status = "Clean"

            for pred in filtered_preds_people:
                # set to default
                overlapped_area = 0 
                max_overlapped_area = 0
                max_area_tb_id = float('-inf')
                
                x_start, y_start, x_end, y_end = get_rectangle_start_end(pred, shape)
                ppl_bbox = ((x_start, y_start), (x_end, y_end))
                
                for id, table in enumerate(table_list):
                    if not overlapped(table.bbox, ppl_bbox):
                        continue

                    overlapped_area = overlapped_area_cal(table.bbox, ppl_bbox)
                    if overlapped_area > max_overlapped_area:
                        max_overlapped_area = overlapped_area
                        max_area_tb_id = table.id

                # update the table status to Occupied if the bbox of the table has the maximum overlapped area with this person 
                for id, table in enumerate(table_list):
                    if max_area_tb_id == table.id:
                        occupied.add(id)
            
            for id, table in enumerate(table_list):
                if id in occupied:
                    table.counter += 1
                
                else:
                    table.counter = 0

                if table.counter >= ol_t_threshold: # more than three sec
                    table.status = 'Occupied'
                    table.last_overlapped_ts = now

            # update the table status 
            for id, table in enumerate(table_list):
                # consider the case if people leave for a while but haven't complete dinning
                if table.status != 'Occupied' and table.last_overlapped_ts:
                    # time difference calculation
                    delta = now - table.last_overlapped_ts
                    if delta <= timedelta(seconds=lv_t_threshold):
                        table.status = 'Occupied'
                        continue

                    crop_image(table.bbox[0], table.bbox[1], table.bbox[2], table.bbox[3], img, i, table.id)
                    is_table_clean = detect_table_cleanliness(i, table.id)
                    
                    if is_table_clean:
                        table.status = 'Clean'
                    else:
                        table.status = 'Need_clean'

            # update the color of the table
            img = update_tb_color_by_status(img, overlay, table_list)
            
            # draw the bbox of the people 
            for pred in filtered_preds_people:               
                x_start, y_start, x_end, y_end = get_rectangle_start_end(pred, shape)
                ppl_bbox = ((x_start, y_start), (x_end, y_end))
                img = cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (255, 128, 0), 3)
                img = cv2.putText(img, pred.tag_name, (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 3) 
          
        cv2.imwrite(f"{tagged_folder}/tagged{i:04d}.jpg", img)
    
    print('5. Finish inferencing the frames of the video')


def compose_video(frame_count, PUBLISH_ITERATION_NAME, prob_threshold, project_id):
    tagged_folder = f'{project_id}_tagged_images'
    video_name = f'tagged_{PUBLISH_ITERATION_NAME}_{prob_threshold}.avi'
    img = cv2.imread(f'{project_id}_tagged_images/tagged0180.jpg')
    shape = img.shape

    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out_video = cv2.VideoWriter(video_name, fourcc, 25, (shape[1], shape[0]))

    for i in range(frame_count):
        file_name = f'{tagged_folder}/tagged{i:04d}.jpg'
        img = cv2.imread(file_name)
        out_video.write(img)
        
    out_video.release()

    print('6. Finish composing the video')
    print(f'7. Check the inferenced video - {video_name} under the ai-inerence-on-top-of-video folder')

if __name__ == "__main__":
    main()


