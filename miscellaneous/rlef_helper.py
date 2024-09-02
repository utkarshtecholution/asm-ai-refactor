import requests
import json
import random
import Config
import os




def multipointPath_To_WeirdAutoaiAnnotationFormat(annotations, label):
    li = {}
    # obj = "label"
    
    for idx, ann in enumerate(annotations):
        # xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

        li[f"[[{ann[0][0]}, {ann[0][1]}], [{ann[1][0]}, {ann[1][1]}], [{ann[3][0]}, {ann[3][1]}], [{ann[2][0]}, {ann[2][1]}]]"] = label[idx]   ## CHANGE IT LATER ACCORDINGLY
        # obj = bbox[4]
    
    rlef_format = json_creater(li, True)
    # print(rlef_format)
    return rlef_format

def segmentation_annotation_rlef(segments, label):
    li = {}

    for idx, segment in enumerate(segments):
        li[str(segment)] = label[idx]
    rlef_format = json_creater(li,True)
    return rlef_format

def pointPath_To_WeirdAutoaiAnnotationFormat(bboxes, labels):
    li = {}
    # obj = "label"
    for bbox, label in zip(bboxes, labels):
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

        li[f"[[{xmin}, {ymin}], [{xmin}, {ymax}], [{xmax}, {ymax}], [{xmax}, {ymin}]]"] = label   ## CHANGE IT LATER ACCORDINGLY
        # obj = bbox[4]

    rlef_format = json_creater(li, True)
    return rlef_format
       
def json_creater(inputs, closed):
    data = []
    count = 1
    highContrastingColors = ['rgba(0,255,81,1)', 'rgba(255,219,0,1)', 'rgba(255,0,0,1)', 'rgba(0,4,255,1)',
                             'rgba(227,0,255,1)']
    for index, input in enumerate(inputs):
        color = random.sample(highContrastingColors, 1)[0]
        json_id = count
        sub_json_data = {}
        sub_json_data['id'] = json_id
        sub_json_data['name'] = json_id
        sub_json_data['color'] = color
        sub_json_data['isClosed'] = closed
        sub_json_data['selectedOptions'] = [{"id": "0", "value": "root"},
                                            {"id": str(random.randint(10, 20)), "value": inputs[input]}]
        points = eval(input)

        vertices = []
        is_first = True
        for vertex in points:
            vertex_json = {}
            if is_first:
                vertex_json['id'] = json_id
                vertex_json['name'] = json_id
                is_first = False
            else:
                json_id = count
                vertex_json['id'] = json_id
                vertex_json['name'] = json_id
            vertex_json['x'] = vertex[0]
            vertex_json['y'] = vertex[1]
            vertices.append(vertex_json)
            count += 1
        sub_json_data['vertices'] = vertices
        data.append(sub_json_data)
    return json.dumps(data)

def increment_version(version):
    parts = version.split('.')
    if len(parts) != 3:
        raise ValueError("Invalid version format. Must be in the format 'x.y.z'")

    major, minor, patch = map(int, parts)
    patch += 1
    return f"{major}.{minor}.{patch}"

def get_biggest_version(model_id, model_arch):
    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/collection/model/biggestVersion"
    payload = {
        "modelId": model_id,
        "modelArchitecture": model_arch
    }
    response = requests.get(url, payload)
    return increment_version(response.json())

def train_model(model_id, collection_name, collection_id, model_arch, training):
    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/collection/model"
    train_col = "trainingDataSetCollections" if training else "reTrainingDataSetCollections"
    print(f'{collection_id} is sent to train_model function')
    print(f'{Config.MASTER_COLLECTION_ID} is the MASTER COLLECTION ID')
    payload = json.dumps({
        "model": model_id,
        "trainingDataDirectory": collection_name,
        "modelArchitecture": model_arch,
        "modelDescription": f"Training with {collection_name}",
        "startCheckpoint": "best.pt",
        "hyperParameter": json.dumps(Config.HP[model_arch]),
        "version": get_biggest_version(model_id, model_arch),
        "parentCheckpointFileData": None,
        "status": "Queued",
        "scenario": train_col,
        "defaultDataSetCollectionId": collection_id
    })
    response = requests.request("POST", url, headers=Config.HEADERS, data=payload)
    if response.status_code == 200:
        print(response.json())
    else:
        return 400
    
    

def check_if_collection_not_exists(model_id, name):
    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/collection/dataSet/checkDatasetCollectionNameExist"
    payload = {
        "modelId": model_id,
        "datasetCollectionName": name
    }
    response = requests.get(url, payload)
    if response.json()["unique"]:
        return None
    else:
        print(response.json()["existingDataSetCollectionId"])
        return response.json()["existingDataSetCollectionId"]
    

def create_collection(model_id, name):
    collection_id = check_if_collection_not_exists(model_id, name)
    if collection_id:
        return collection_id
    
    print("Creating Collection")
    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/collection/dataSet"

    payload = {
      "name": name,
      "model": model_id,
      "description": "Data-set collection created for testing purpose"
    }
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload)
    collection_id = response.json()["_id"]

    return collection_id

def add_resource_to_collection(collection_id, resource_ids):
    print("Adding Resources to Collection")
    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/collection/dataSet/addResource"
    payload = {
      "dataSetCollectionIds": [collection_id],
      "resourceIds": resource_ids
    }
    # print(payload)
    response = requests.put(url, json=payload)
    print(response.status_code)

def add_collection_to_master(master_id, collection_id):
    print("Adding Collection to Master")
    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/collection/dataSet/addDataSetCollection"
    payload = {
      "childDatasetCollectionId": collection_id,
      "parentDatasetCollectionId": master_id
    }
    response = requests.put(url, json=payload)
    print(response.status_code)
    
def remove_collection_from_master(master_id, collection_id):
    print("Removing Collection from Master")
    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/collection/dataSet/removeDataSetCollection"
    payload = {
      "childDatasetCollectionId": collection_id,
      "parentDatasetCollectionId": master_id
    }
    response = requests.put(url, json=payload)
    print(response.status_code)
    if response.status_code == 200:
        return True, 200
    else:
        return False, 400
    
    
    
def get_all_collections(model_id):
    print("Print all the collection")
    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/collection/dataSet"
    payload = {
        "model": model_id,
        "limit": -1,
        "offset": -1
    }
    response = requests.get(url, payload)
    return response.json()
    
    
def send_to_rlef(img_path, model_id, tag,label, status, annotation, confidence_score=100, prediction='predicted'):
    print("Sending")
    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource"
    payload = {
        'model': model_id,
        'status': status,
        'csv': 'csv',
        'label': label,
        'tag': tag,
        'model_type': 'imageAnnotation',
        'prediction': prediction,
        'confidence_score': confidence_score
    }
    if annotation:
        payload["imageAnnotations"] = str(annotation)
    if os.path.exists(img_path):
        print('Image exists')
    
    files = [('resource', (f'{img_path}', open((img_path), 'rb'), 'image/png'))]
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    print(response.status_code)

#     if response.status_code == 200:
#         resource_id = response.json()["_id"]
#         return resource_id
    
#     exceptions = []
#     if response.status_code != 200:
#         raise Exception(f'ERROR UPLOADING ORIGINAL IMAGE: {response.status_code}')
#         exceptions.append(img_path)
#         print('code: ', response.status_code)
#         return exceptions
    
