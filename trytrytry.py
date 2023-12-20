import cv2
import numpy as np
import mediapipe as mp
import sys

width, height = (500, 500)
cap = cv2.VideoCapture("video.mp4")
result1 = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 25, (500, 500))

cap.set(3, width)
cap.set(4, height)


image = cv2.imread("hehe.jpg")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def get_landmark_posistion(src_image):
    with mp_face_mesh.FaceMesh(
            static_image_mode= True,
            max_num_faces= 1,
            refine_landmarks= True,
            min_detection_confidence= 0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return None
        
        if len(results.multi_face_landmarks > 1):
            sys.exit("There are too many face landmarks.")

        src_face_landmark = results.multi_face_landmarks[0].landmark
        landmark_points = []

        for i in range(468):
            y = int(src_face_landmark[i].y * src_image.shape[0])
            x = int(src_face_landmark[i].x * src_image.shape[1])
            landmark_points.append((x, y))

            return landmark_points
        
def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def get_triangles(convexhull, landmark_points, np_points):
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmark_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    
    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((np_points == pt1).all(axis = 1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((np_points == pt2).all(axis = 1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((np_points == pt3).all(axis = 1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

    return indexes_triangles


def triangulation(triangle_index, landmark_points, img=None):
    tr1_pt1 = landmark_points[triangle_index[0]]
    tr1_pt2 = landmark_points[triangle_index[1]]
    tr1_pt3 = landmark_points[triangle_index[2]]
    
    rect = cv2.boundingrect(triangle)
    (x, y, w, h) = rect

    cropped_triangle = None
    if img is not None:
        cropped_triangle = img[y:y + h, x:x + w]

    cropped_triangle_mask = np.zeros((h, w), np.uint8)
    points = np.array([tr1_pt1[0] - x, tr1_pt1[1] - y],
                     [tr1_pt2[0] - x, tr1_pt2[1] - y],
                     [tr1_pt3[0] - x, tr1_pt3[1] - y], np.uint32)

    cv2.fillConvexPoly(cropped_triangle_mask, points, 255)
    return points, cropped_triangle, cropped_triangle_mask, rect

def warped_triangle(rect, points1, points2, src_cropped_triangle, dest_cropped_triangle_mask):
    (x, y, w, h) = rect

    matrix = cv2. getAffineTransform(np.float32(points1), np.float32(points2))
    warped_triangle = cv2.warpAffine(src_cropped_triangle, matrix, (w, h))
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask= dest_cropped_triangle_mask)

    return warped_triangle

def add_piece_of_new_face(new_face, rect, warped_triangle):
    (x, y, w, h) = rect
    new_face_rect_area = new_face[y:y + h, x:x + w]
    new_face_rect_area_gray = cv2.cvtColor(new_face_rect_area, cv2.COLOR_BGR2GRAY)

    _, mask_triangles_designed = cv2.threshold(new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)

    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask= mask_triangles_designed)

    new_face_rect_area = cv2.add(new_face_rect_area, warped_triangle)
    new_face[y:y + h, x:x + w] = new_face_rect_area

def swap_new_face(dest_image, dest_image_gray, dest_convexHull, new_face):
    face_mask = np.zeros_like(dest_image_gray)
    head_mask = cv2.fillconvexPoly(face_mask, dest_convexHull, 255)
    face_mask = cv2.bitwise_not(head_mask)

    head_without_face = cv2.bitwise_and(dest_image, dest_image, mask= face_mask)

    result = cv2.add(head_without_face, new_face)

    (x, y, w, h) = cv2.boundingRect(dest_convexHull)
    center_face = (int((x + x + w) / 2), int((y + y + h) / 2))

    return cv2.seamlessClone(result, dest_image, head_mask, center_face, cv2.MIXED_CLONE)

