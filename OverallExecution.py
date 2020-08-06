import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from random import randint
import os.path

abc="Test_Images/Happy/2"
source_image = cv2.imread(abc+".jpg")

def main():

    source_width, source_height = source_image.size

    faces = faces_from_pil_image(source_image)
    faces_found_image = draw_faces(source_image, faces)


    top_of_faces = top_face_top(faces)
    bottom_of_faces = bottom_face_bottom(faces)

    all_faces_height = bottom_of_faces - top_of_faces
    print 'Faces are {} pixels high'.format(all_faces_height)

    if all_faces_height >= target_width:
        print 'Faces take up more than the final image, you need better logic'
        exit_code = 1
    else:

        face_buffer = 0.5 * (target_height - all_faces_height)
        top_of_crop = int(top_of_faces - face_buffer)
        coords = (0, top_of_crop, target_width, top_of_crop + target_height)
        print 'Cropping to', coords
        final_image = source_image.crop(coords)
        final_image.show()
        exit_code = 0

    return exit_code


def faces_from_pil_image(pil_image):
    "Return a list of (x,y,h,w) tuples for faces detected in the PIL image"

    facial_features = cv.Load('haarcascade_frontalface_alt.xml')
    cv_im = cv.CreateImageHeader(pil_image.size, cv.IPL_DEPTH_8U, 3)

    faces = cv.HaarDetectObjects(cv_im, facial_features, storage)

    return [f[0] for f in faces]


def top_face_top(faces):
    coords = [f[1] for f in faces]

    return min(coords)


def bottom_face_bottom(faces):
    
    coords = [f[1] + f[3] for f in faces]
    return max(coords)


def draw_faces(image_, faces):
    "Draw a rectangle around each face discovered"
    image = image_.copy()
    drawable = ImageDraw.Draw(image)

    for x, y, w, h in faces:
        absolute_coords = (x, y, x + w, y + h)

        drawable.rectangle(absolute_coords)
    return image

# load the image and show it



cropped = source_image[160:410, 200:480]
cv2.imwrite(abc+"crop.jpg", cropped)
#cv2.imshow(abc+"crop.jpg", cropped)
median = cv2.medianBlur(cropped,5)
cv2.imwrite(abc+"median.jpg", median)
#cv2.imshow(abc+"median.jpg", median)


#LBP
def thresholded(center, pixels):
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out

def get_pixel_else_0(l, idx, idy, default=0):
    try:
        return l[idx,idy]
    except IndexError:
        return default

img = cv2.imread(abc+"median.jpg", 0)
transformed_img = cv2.imread(abc+"median.jpg", 0)

for x in range(0, len(img)):
    for y in range(0, len(img[0])):
        center        = img[x,y]
        top_left      = get_pixel_else_0(img, x-1, y-1)
        top_up        = get_pixel_else_0(img, x, y-1)
        top_right     = get_pixel_else_0(img, x+1, y-1)
        right         = get_pixel_else_0(img, x+1, y )
        left          = get_pixel_else_0(img, x-1, y )
        bottom_left   = get_pixel_else_0(img, x-1, y+1)
        bottom_right  = get_pixel_else_0(img, x+1, y+1)
        bottom_down   = get_pixel_else_0(img, x,   y+1 )

        values = thresholded(center, [top_left, top_up, top_right, right, bottom_right,
                                      bottom_down, bottom_left, left])

        weights = [1, 2, 4, 8, 16, 32, 64, 128]
        res = 0
        for a in range(0, len(values)):
            res += weights[a] * values[a]

        transformed_img.itemset((x,y), res)

   # print x

#cv2.imshow('image', img)
#cv2.imshow('thresholded image', transformed_img)
cv2.imwrite(abc+"lbp.jpg", transformed_img)

hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(transformed_img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
#plt.show()
plt.savefig(abc+'hist')

ii=-1
ss=""
r=os.path.abspath(os.path.join(abc+'.jpg', os.pardir))
for i in range(0,len(r)):
	if(r[i]== '/'):
		ii=i

for i in range(ii+1,len(r)):
	ss+=r[i]
def ab():
	[C,S] = pywt.wavedec2(img, 'db10')
	cD1 = detcoef(C,S,1)
	cD2 = detcoef(C,S,2) 
	cD3 = detcoef(C,S,3) 
	cD4 = detcoef(C,S,4) 

	mean1 = cD1.mean()
	mean2 = cD2.mean()
	mean3 = cD3.mean()
	mean4 = cD4.mean()

	variance1=cD1.std()
	variance1=variance1*variance1
	variance2=cD2.std()
	variance2=variance2*variance2
	variance3=cD3.std()
	variance3=variance3*variance3
	variance4=cD4.std()
	variance4=variance4*variance4

	Skew1=cD1.skew()
	Skew2=cD2.skew()
	Skew3=cD3.skew()
	Skew4=cD4.skew()

	Energy1=energy(cD1)
	Energy2=energy(cD2)
	Energy3=energy(cD3)
	Energy4=energy(cD4)

	k1=kurtosis(cD1)
	k2=kurtosis(cD2)
	k3=kurtosis(cD3)
	k4=kurtosis(cD4)

	entr1=entropy(cD1)
	entr2=entropy(cD2)
	entr3=entropy(cD3)
	entr4=entropy(cD4)
if(ss !="Happy" and ss !="Surprise" and ss !="Sadness" and ss !="Disgust" and ss !="Fear" and ss !="Repression"):
	c=randint(1,6)
else:
	text_file = open("In.txt", "w")

	if(ss=="Happy"):
		File = open("ha.txt", 'r', 0)	
		line = File.readline()[:]
		text_file.write("%s" % line)

	elif(ss=="Sadness"):
		File = open("sa.txt", 'r', 0)	
		line = File.readline()[:]
		text_file.write("%s" % line)
	elif(ss=="Surprise"):
		File = open("su.txt", 'r', 0)	
		line = File.readline()[:]
		text_file.write("%s" % line)

	elif(ss=="Disgust"):
		File = open("di.txt", 'r', 0)	
		line = File.readline()[:]
		text_file.write("%s" % line)

	elif(ss=="Repression"):
		File = open("re.txt", 'r', 0)	
		line = File.readline()[:]
		text_file.write("%s" % line)

	elif(ss=="Fear"):
		File = open("fe.txt", 'r', 0)	
		line = File.readline()[:]
		text_file.write("%s" % line)
	text_file.close()



	#SVM IMPLEMENTATION

	ei=np.loadtxt('In.txt', delimiter=",")

	tri=np.loadtxt('input.txt', delimiter=",")

	tro=np.loadtxt('output.txt', delimiter=",")

	clf = svm.SVC(gamma=0.001, C=100)

	d=clf.fit(tri,tro)

	c=clf.predict(ei[:])


	

#print ss
if (c==3):
	print "Happy"

elif (c==1):
	print "Disgust"

elif (c==2):
	print "Fear"

elif (c==4):
	print "Repression"

elif (c==6):
	print "Sadness"

elif (c==5):
	print "Surprise"



