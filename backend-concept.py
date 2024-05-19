import cv2
import numpy as np
import os
from PIL import Image
import pillow_heif
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def read_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.heic')):
            img_path = os.path.join(directory, filename)
            if filename.lower().endswith('.heic'):
                img = read_heic_image(img_path)
            else:
                img = cv2.imread(img_path)
            if img is not None:
                images.append((img, filename))
    return images

def read_heic_image(path):
    heif_file = pillow_heif.read_heif(path)
    image = Image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data, 
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def extract_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors

def cluster_images(images, n_clusters):
    all_descriptors = []
    for img, filename in images:
        descriptors = extract_features(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if descriptors is not None:
            all_descriptors.extend(descriptors)
    
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(all_descriptors)
    
    clusters = [[] for _ in range(n_clusters)]
    for img, filename in images:
        descriptors = extract_features(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if descriptors is not None:
            labels = kmeans.predict(descriptors)
            cluster = np.bincount(labels).argmax()
            clusters[cluster].append((img, filename))
    
    return clusters

def calculate_parallax(left_image, right_image):
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(left_gray, right_gray)
    
    parallax = np.sum(np.abs(disparity))
    return parallax

def find_optimal_pair(images):
    min_parallax = float('inf')
    optimal_pair = (None, None)
    
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            parallax = calculate_parallax(images[i], images[j])
            if parallax < min_parallax:
                min_parallax = parallax
                optimal_pair = (images[i], images[j])
    
    return optimal_pair

def determine_order(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    points1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches])

    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
    lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)

    displacement1 = np.median([np.abs(np.dot(line, [x, y, 1])) for line, (x, y) in zip(lines1, points1)])
    displacement2 = np.median([np.abs(np.dot(line, [x, y, 1])) for line, (x, y) in zip(lines2, points2)])

    if displacement1 < displacement2:
        return image1, image2
    else:
        return image2, image1

def align_images(left_image, right_image):
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(left_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(right_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    points1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches])

    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    height, width, channels = left_image.shape
    aligned_right_image = cv2.warpPerspective(right_image, h, (width, height))

    return left_image, aligned_right_image

def crop_to_intersection(left_image, right_image):
    mask_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    mask_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    _, mask_left = cv2.threshold(mask_left, 1, 255, cv2.THRESH_BINARY)
    _, mask_right = cv2.threshold(mask_right, 1, 255, cv2.THRESH_BINARY)

    intersection_mask = cv2.bitwise_and(mask_left, mask_right)
    x, y, w, h = cv2.boundingRect(intersection_mask)

    left_image_cropped = left_image[y:y+h, x:x+w]
    right_image_cropped = right_image[y:y+h, x:x+w]

    return left_image_cropped, right_image_cropped

def create_stereo_image(left_image, right_image):
    height = max(left_image.shape[0], right_image.shape[0])
    width = left_image.shape[1] + right_image.shape[1]
    stereo_image = np.zeros((height, width, 3), dtype=np.uint8)

    stereo_image[:left_image.shape[0], :left_image.shape[1]] = left_image
    stereo_image[:right_image.shape[0], left_image.shape[1]:] = right_image

    return stereo_image

def main(directory, n_clusters=5):
    images = read_images_from_directory(directory)
    if len(images) < 2:
        print("Недостаточно изображений для создания стереопары.")
        return
    
    clustered_images = cluster_images(images, n_clusters)
    
    for idx, cluster in enumerate(clustered_images):
        if len(cluster) < 2:
            print(f"Недостаточно изображений для создания стереопары в кластере {idx}.")
            continue
        
        images_only = [img for img, _ in cluster]
        image1, image2 = find_optimal_pair(images_only)
        if image1 is None or image2 is None:
            print(f"Не удалось найти оптимальную пару изображений в кластере {idx}.")
            continue
        
        left_image, right_image = determine_order(image1, image2)
        left_image, right_image = align_images(left_image, right_image)
        left_image, right_image = determine_order(left_image, right_image)  # Проверка порядка после выравнивания
        left_image, right_image = crop_to_intersection(left_image, right_image)
        stereo_image = create_stereo_image(left_image, right_image)
        stereo_image_filename = os.path.join(directory, f"stereo_{idx}.png")
        
        cv2.imwrite(stereo_image_filename, stereo_image)
        print(f"Стереоизображение сохранено в {stereo_image_filename}")

if __name__ == "__main__":
    directory = './images2/'  # Замените на путь к вашей директории с изображениями
    main(directory)
