import os
import csv

age_list0 = []
image_dir = '../stargan_cefinal/result/age_group0'
age_label = 0
image_list = os.listdir(image_dir)
total_age = 0.0
correct = 0
length = len(image_list)

print(len(image_list))
for i in range(len(image_list)):
    print(i)
    try:
        filename = image_list[i]
        jpgfile = os.path.join(image_dir, filename)
        address = "/home/seung/Desktop/test" + str(i) + ".jpg"
        terminal_order = 'curl -X POST "https://api-us.faceplusplus.com/facepp/v3/detect" -F "api_key=nxChMZYeFq7BCj_Ia7rj6Oii5f8-0LE7" \
        -F "api_secret=m8E-GDmTes_5Jte8_E3ro5Zn5Xfko5nS" \
        -F "image_file=@ ../stargan_cefinal/result/age_group0/' + filename + '" \
        -F "return_landmark=1" \
        -F "return_attributes=age"'


        result = os.popen(terminal_order).read()
        print(result)
        target = '"value":'
        target_pos = result.find('"value":')
        age = str(result[target_pos+8]) + str(result[target_pos+9])
        
        
        age_list0.append(age)
        print(age)
    except:
        break

age_list1 = []
image_dir = '../stargan_cefinal/result/age_group1'
age_label = 1
image_list = os.listdir(image_dir)
total_age = 0.0
correct = 0
length = len(image_list)

print(len(image_list))
for i in range(len(image_list)):
    print(i)
    try:
        filename = image_list[i]
        jpgfile = os.path.join(image_dir, filename)
        address = "/home/seung/Desktop/test" + str(i) + ".jpg"
        terminal_order = 'curl -X POST "https://api-us.faceplusplus.com/facepp/v3/detect" -F "api_key=nxChMZYeFq7BCj_Ia7rj6Oii5f8-0LE7" \
        -F "api_secret=m8E-GDmTes_5Jte8_E3ro5Zn5Xfko5nS" \
        -F "image_file=@ ../stargan_cefinal/result/age_group1/' + filename + '" \
        -F "return_landmark=1" \
        -F "return_attributes=age"'


        result = os.popen(terminal_order).read()
        print(result)
        target = '"value":'
        target_pos = result.find('"value":')
        age = str(result[target_pos+8]) + str(result[target_pos+9])
        
        
        age_list1.append(age)
        print(age)
    except:
        break

age_list2 = []
image_dir = '../stargan_cefinal/result/age_group2'
age_label = 2
image_list = os.listdir(image_dir)
total_age = 0.0
correct = 0
length = len(image_list)

print(len(image_list))
for i in range(len(image_list)):
    print(i)
    try:
        filename = image_list[i]
        jpgfile = os.path.join(image_dir, filename)
        address = "/home/seung/Desktop/test" + str(i) + ".jpg"
        terminal_order = 'curl -X POST "https://api-us.faceplusplus.com/facepp/v3/detect" -F "api_key=nxChMZYeFq7BCj_Ia7rj6Oii5f8-0LE7" \
        -F "api_secret=m8E-GDmTes_5Jte8_E3ro5Zn5Xfko5nS" \
        -F "image_file=@ ../stargan_cefinal/result/age_group2/' + filename + '" \
        -F "return_landmark=1" \
        -F "return_attributes=age"'


        result = os.popen(terminal_order).read()
        print(result)
        target = '"value":'
        target_pos = result.find('"value":')
        age = str(result[target_pos+8]) + str(result[target_pos+9])
        
        
        age_list2.append(age)
        print(age)
    except:
        break


age_list3 = []
image_dir = '../stargan_cefinal/result/age_group3'
age_label = 3
image_list = os.listdir(image_dir)
total_age = 0.0
correct = 0
length = len(image_list)

print(len(image_list))
for i in range(len(image_list)):
    print(i)
    try:
        filename = image_list[i]
        jpgfile = os.path.join(image_dir, filename)
        address = "/home/seung/Desktop/test" + str(i) + ".jpg"
        terminal_order = 'curl -X POST "https://api-us.faceplusplus.com/facepp/v3/detect" -F "api_key=nxChMZYeFq7BCj_Ia7rj6Oii5f8-0LE7" \
        -F "api_secret=m8E-GDmTes_5Jte8_E3ro5Zn5Xfko5nS" \
        -F "image_file=@ ../stargan_cefinal/result/age_group3/' + filename + '" \
        -F "return_landmark=1" \
        -F "return_attributes=age"'


        result = os.popen(terminal_order).read()
        print(result)
        target = '"value":'
        target_pos = result.find('"value":')
        age = str(result[target_pos+8]) + str(result[target_pos+9])
        
        
        age_list3.append(age)
        print(age)
    except:
        break

with open('ce_age.csv', 'w', newline='') as f: 
    writer = csv.writer(f) 
    writer.writerow(age_list0) 
    writer.writerow(age_list1) 
    writer.writerow(age_list2) 
    writer.writerow(age_list3) 
