import firebase_admin
from firebase_admin import db, credentials
import uuid
import datetime
import Config 
from Config import *
from threading import Thread
from google.cloud import storage
from queue import Queue
from thefuzz import fuzz 
import json 
import uuid 
import time 
import traceback 


# Global variables
qsite_queue = Queue()
image_queue = Queue()
onboard_trigger_map = {
    "djkfia": False,
    "askdfi": False,
    "rackReference": False,
    "61c03eef-840c-49df-ba20-0952553a4953": False
}

# Initialize Firebase and Google Cloud Storage
def initialize_firebase_and_storage():
    global bucket, storage_client, bucket_name, app

    storage_client = storage.Client.from_service_account_json("firebase_keys/owens-and-minor-inventory-595f638304db.json")
    bucket_name = 'owm_application_sink'
    bucket = storage_client.bucket(bucket_name)

    firebase_path = "firebase_keys/owens-and-minor-inventory-firebase-adminsdk-hldq7-c3e53eff9c.json"
    databaseURL = "https://owens-and-minor-inventory-onboarding-poc.asia-southeast1.firebasedatabase.app/"
    creds = credentials.Certificate(firebase_path)
    app = firebase_admin.initialize_app(creds, {'databaseURL': databaseURL})

# Upload image and generate signed URL
def upload_and_generate_signed_url(image_path, queue_object, type = None, multi_threaded = False):
    print(f'Type : {type} ----- Imagepath : {image_path}')
    try:
        s = time.time()
        global crop_url
        blob_name = f"onboard_flow/{uuid.uuid1()}_{image_path}.png"
        blob = bucket.blob(blob_name)
    
        blob.upload_from_filename(image_path)
        # print("File Uploaded")
        blob = bucket.blob(blob_name)

        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(days=7),
            method="GET",
        )

        if multi_threaded:
            print(f'Generated Signed URL for type : {type}')
            image_data = {"url" : url, 
                        "type" : type
                        }
            queue_object.put(image_data)
            return 

        # if queue_object is not No÷
        e = time.time()
        print(f'{e-s} for uploading one image onto Google cloud')
        return url
    except Exception as e:
        traceback.print_exc()
# Firebase handler for monitoring compartment "djkfia"
def firebase_handler1(message):
    global crop_url
    if message.data == True:
        onboard_trigger_map["djkfia"] = True
        ref = db.reference(f'/planogram/djkfia/captured_image')
        ref.set(crop_url)
        ref = db.reference('new_product_onboarding/')
        ref.set({
            'GTIN': '00350770954875',
            'assigned_compartment': 'djkfia',
            'barcode': '',
            'captured_image': crop_url,
            'product_name': 'Pilot Guide Wire',
            'ref_no': '1010480-H',
            'update_id': str(uuid.uuid4())
        })

        ref = db.reference('/planogram/djkfia/item_added')
        ref.set(True)
    print(message.data)

# Firebase handler for monitoring compartment "askdfi"
def firebase_handler2(message):
    global crop_url
    if message.data == True:
        onboard_trigger_map["askdfi"] = True
        ref = db.reference(f'/planogram/askdfi/captured_image')
        ref.set(crop_url)
        ref = db.reference('new_product_onboarding/')
        ref.set({
            'GTIN': '00350770954875',
            'assigned_compartment': 'askdfi',
            'barcode': '',
            'captured_image': crop_url,
            'product_name': 'Pilot Guide Wire',
            'ref_no': '1010480-H',
            'update_id': str(uuid.uuid4())
        })
        ref = db.reference('/planogram/askdfi/item_added')
        ref.set(True)
    print(message.data)

# Start monitoring for compartment "djkfia"
def shelf_monitoring1():
    ref = db.reference('/planogram/djkfia/onboarding_started')
    ref.listen(firebase_handler1)

# Start monitoring for compartment "askdfi"
def shelf_monitoring2():
    ref = db.reference('/planogram/askdfi/onboarding_started')
    ref.listen(firebase_handler2)

# Early call for uploading image and updating database
def early_call(image_path, compartment_id):
    print('>>>>>>>>>>>>>>> EARLY CALL <<<<<<<<<<<')
    url = upload_and_generate_signed_url(image_path, compartment_id, image_queue)
    assign_ref = db.reference(f'/planogram/{compartment_id}/captured_image')
    assign_ref.set(url)
    ref = db.reference(f'/planogram/{compartment_id}/item_added')
    ref.set(True)
    print('DUMMY')
    Config.DUMMY_DB['product_image'] = url
    Config.DUMMY_DB['captured_image'] = url

    ref = db.reference(f'/new_product_onboarding')
    ref.update(Config.DUMMY_DB)

    

# Update action status in database
def action_status_update(action, compartment_id):
    if action == "Grab":
        key_to_update = "item_grabbed"
    elif action == "misplaced":
        key_to_update = "item_misplaced"
    else:
        key_to_update = "item_returned"

    ref = db.reference(f'/planogram/{compartment_id}/{key_to_update}')
    ref.set(True)




def update_runtime_image(compartment_id, image_path):
    ref = db.reference(f'/planogram/{compartment_id}/captured_image')
    url = upload_and_generate_signed_url(image_path, compartment_id, None)
    ref.set(url)
# Update misplaced status in database

def misplace_status_update(compartment_id, image_path, ref_no, status):
    print(compartment_id)
    ref = db.reference(f'/planogram/{compartment_id}')
    url = upload_and_generate_signed_url(image_path, compartment_id, image_queue)

    data = ref.get()
    try:
        if status is True:
            count_ref = db.reference(f'/planogram/{compartment_id}/assigned_product/item_count')
            count_ref.set(count_ref.get() - 1)
    except Exception as e:
        print(e)

    print('NEW REF NO:', ref_no)
    data['item_misplaced'] = status
    data['capture_image_ref_id'] = ref_no
    data['captured_image'] = url
    ref.update(data)

# Get onboarding status
def get_onboard_status(compartment_id):
    ref = db.reference(f'/planogram/{compartment_id}/onboarding_started')
    return ref.get()
    # onboarding_status = []
    # for compartment_id in compartment_list:
    #     ref = db.reference(f'/planogram/{compartment_id}/onboarding_started')
    #     onboarding_status.append(ref.get())
    # return onboarding_status

# Update onboarded details in the database
def update_onboarded_details(compartment_id, image_path, sku_details):
    print('>>>>UPDATE ONBOARDING <<<')

    def upload_images():
        qsite_url = upload_and_generate_signed_url("1010480-H.png", compartment_id, qsite_queue, False)
        url = upload_and_generate_signed_url(image_path, compartment_id, image_queue, False)
        return qsite_url, url

    th1 = Thread(target=upload_images)
    th1.start()
    th1.join()

    qsite_url, url = qsite_queue.get(), image_queue.get()
    print(f"URL: {url}")

    ref = db.reference(f'/new_product_onboarding')
    sku_details['captured_image'] = url
    sku_details['item_count'] = 1
    sku_details['Qsight_image'] = qsite_url
    sku_details['assigned_compartment'] = compartment_id
    print(compartment_id)
    sku_details['product_image'] = url

    print('CODE REACHED')
    print('<<<<<<<<<>>>>>>>>>>>>>>>>')

    ref.update(sku_details)
    # Actual Database update happens here

    ref = db.reference(f'/planogram/{compartment_id}/onboarding_started')
    ref.set(False)






def update_workstation_details(sku_details) -> dict:
    ref = db.reference('/ai_workstation_status')
    product_id = get_product_unique_id(sku_details['ref_no'])
    status_data = {"product_detected" :product_id ,
                   "current_status" : "detected"
    }


   
    ref.set(status_data)
  
    try:
        product_details = db.reference(f'Products/{product_id}').get()
    except Exception as e:
        product_details = {}

    ref = db.reference(f'/procedure_logs/{product_id}')
    
    if sku_details['ref_no'] == '':
        sku_details['grabbed_at'] = "12:00pm"
        sku_details["is_scanning"] = False 
        sku_details["item_consumed"] = False 
        sku_details["item_discarded"] = False 
        sku_details["item_expired"] = False 
        sku_details["item_intransit"] = False 
        sku_details["item_returned"] = False 
        sku_details["product_description"] = '59 long tapered'
        sku_details["product_type"] = "Bipolar Atrial"
        sku_details["quantity"] = 1
        sku_details["is_documented"] = False
        ref.update(sku_details)
        return 
    
    if product_details != {}:
        keys = ["product_image", "procedure", "product_description", 
                "product_type", "catalog_image"
                ]
        # sku_details["product_image"] = product_details["product_image"]
        sku_details["procedure"] = product_details["procedure"]
        sku_details["product_description"] = product_details["product_description"]
        sku_details["product_type"] = product_details["product_type"]
        sku_details["catalog_image"] = product_details["catalog_image"]
        sku_details['patient_mrn'] = product_details['patient_mrn']
        if 'manufacturer' in product_details:
            sku_details['manufacturer'] = product_details['manufacturer']
        if 'product_name' in product_details:
            sku_details['product_name'] = product_details['product_name']
        
        
    

    ## updating the details of that product unique id only.....
    
    ref.set(sku_details)
    print('done updating firebase')

def update_status_workstation_ui(product_id):
    ref = db.reference('/ai_workstation_status')

    status_data = {"product_detected" :str(product_id) ,
                   "current_status" : "scanning"
    }
    ref.set(status_data)



def match_substring(decoded_text, local_database):

    ## This will make sure that the given product details are present in the database......

    ## Time Complexity : O(N)^2

    max_similarity = 0
    words = decoded_text.split(' ')
    final_ref_no = None 
    for word in words:
        for ref_no in local_database.keys():
            sim_score = fuzz.ratio(word,ref_no)
            if sim_score > 95:
                return ref_no 
            
            if sim_score > max_similarity:
                max_similarity = sim_score 
                final_ref_no = ref_no 

    return final_ref_no 

    

def get_product_unique_id(reference_number):
    with open('cache_database.json', 'r') as json_file:
        local_database = json.load(json_file) 
    
    try:
        unique_id = local_database[reference_number] 
        return unique_id 
    except Exception as e:
        print(f'>>>>>>>>>>> The exact reference number is not found <<<<<<<<<<<<<<<<<<<')
        ## Fallback condition to check which is the closest reference number OCR Decoded
        nearest_reference_no = match_substring(reference_number, local_database)
        if nearest_reference_no is None:
            return str(uuid.uuid1())
        return local_database[nearest_reference_no] 
        


    

# Initialize Firebase and Storage on module import
initialize_firebase_and_storage()

