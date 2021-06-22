import os

age_list = []
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
        
        
        age_list.append(age)
        print(age)
    except:
        break



for i in range(len(age_list)):
    label = 0
    age = age_list[i]
    try:
        age = int(age)
        if age < 14:
            label = -1
        elif age >= 14 and age < 26:
            label = 0
        elif age >= 26 and age < 38:
            label = 1
        elif age >= 38 and age < 50:
            label = 2
        elif age >= 50 and age <= 62:
            label = 3
        elif age > 62:
            label = 4

        if label == age_label:
            correct += 1 
        
        total_age += age

    except:
        print(age)
        length -= 1

print("Average age: ", total_age / length)
print("Classification accuracy:", (correct / length)*100 )
# print(age_list)
