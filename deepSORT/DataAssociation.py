#This skeleton code's structure comes from the research paper: https://www.mdpi.com/2076-3417/12/3/1319

#Contain the CNN that deepSORT uses in replace of the Hungarion based Cost Matrix
from filterpy.kalman import KalmanFilter

class DataAssociation:
    import numpy as np
    from scipy.spatial.distance import cosine

    #Euclidean Distance Based Cost Matrix (𝐷𝐸(𝐷,𝑃))
    def euclidean_cost(tracks, detections, image_dims):
        """
        Computes the Euclidean distance cost matrix (𝐷𝐸(𝐷,𝑃)), which represents
        the distance between bounding box central points normalized into half
        of the image dimension. To formulate the problem as a maximization
        problem, the distance is obtained by the difference between 1 and the
        normalized Euclidean distance.

        d(Di, Pi) = 1 - sqrt((u_Di - u_Pi)^2 + (v_Di - v_Pi)^2) / sqrt(1/2 * (h^2 + w^2))

        where (h, w) are the height and width of the input image.
        """
        pass

    #Bounding Box Ratio Based Cost Matrix (𝑅(𝐷,𝑃))
    def bbox_ratio_cost(tracks, detections):
        """
        Computes the bounding box ratio-based cost matrix (𝑅(𝐷,𝑃)), which is
        implemented as a ratio between the product of each width and height.

        r(Di, Pi) = min( (w_Di * h_Di) / (w_Pi * h_Pi), (w_Pi * h_Pi) / (w_Di * h_Di) )

        This metric gives values closer to 1 for similar box shapes and values
        closer to 0 for significantly different boxes.
        """
        pass

   #SORT’s IoU Cost Matrix
    def iou_cost(tracks,detections):
        
        numDetections = len(detections)
        numTracks = len(tracks)
        
        
        det_x1 = detections[:, 0:1]
        det_y1 = detections[:, 1:2]
        det_x2 = det_x1 + detections[:, 2:3]
        det_y2 = det_y1 + detections[:, 3:4]

        trk_x1 = tracks[:, 0]
        trk_y1 = tracks[:, 1]
        trk_x2 = trk_x1 + tracks[:, 2]
        trk_y2 = trk_y1 + tracks[:, 3]
            
        detectionWidth = detections[:,2:3]
        detectionHeight = detections[:,3:4]
        trackWidth = tracks[:,2]
        trackHeight = tracks[:,3]
               
        areaDetection = detectionWidth * detectionHeight
        areaTrack = trackWidth * trackHeight  

        inter_x1 = np.maximum(det_x1, trk_x1)
        inter_y1 = np.maximum(det_y1, trk_y1)
        inter_x2 = np.minimum(det_x2, trk_x2)
        inter_y2 = np.minimum(det_y2, trk_y2)

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        
 
        
        "AoI = Area of Intersection, AoU = Area of Union"
        AoI = inter_w * inter_h
        AoU = areaDetection + areaTrack - AoI
        union_area = areaDetection + areaTrack - AoI
        iou_matrix = np.where(union_area > 0, AoI / AoU, 0.0)

        

        
       
        return iou_matrix

    #SORT’s IoU Cost Matrix Combined with the Euclidean Distance Cost Matrix (𝐸𝐼𝑜𝑈𝐷(𝐷,𝑃))
    def iou_euclidean_cost(tracks, detections, image_dims):
        """
        Computes the IoU cost matrix combined with the Euclidean distance cost
        matrix using the Hadamard (element-wise) product:

        EIoUD(D, P) = IoU(D, P) ∘ DE(D, P)

        where ∘ represents element-wise multiplication.
        """
        pass

    #SORT’s IoU Cost Matrix Combined with the Bounding Box Ratio Based Cost Matrix (𝑅𝐼𝑜𝑈(𝐷,𝑃))
    def iou_bbox_ratio_cost(tracks, detections):
        """
        Computes the IoU cost matrix combined with the bounding box ratio-based
        cost matrix using the Hadamard (element-wise) product:

        RIoU(D, P) = IoU(D, P) ∘ R(D, P)

        where ∘ represents element-wise multiplication.
        """
        pass

    #Euclidean Distance Cost Matrix Combined with the Bounding Box Ratio Based Cost Matrix (𝑅𝐷𝐸(𝐷,𝑃))
    def euclidean_bbox_ratio_cost(tracks, detections, image_dims):
        """
        Computes the Euclidean distance cost matrix combined with the bounding box
        ratio-based cost matrix using the Hadamard (element-wise) product:

        RDE(D, P) = DE(D, P) ∘ R(D, P)

        where ∘ represents element-wise multiplication.
        """
        pass

    #Step 7: SORT’s IoU Cost Matrix Combined with the Euclidean Distance Cost Matrix and the Bounding Box Ratio Based Cost Matrix (𝑀(𝐷,𝑃))
    def combined_cost_matrix(tracks, detections, image_dims):
        """
        Computes the IoU cost matrix combined with the Euclidean distance cost
        matrix and the bounding box ratio-based cost matrix using the Hadamard
        (element-wise) product:

        M(D, P) = IoU(D, P) ∘ DE(D, P) ∘ R(D, P)

        where ∘ represents element-wise multiplication.
        """
        pass

    #Element-wise Average of Every Cost Matrix (𝐴(𝐷,𝑃))
    def average_cost_matrix(tracks, detections, image_dims):
        """
        Computes the element-wise average of every cost matrix:

        A(Di, Pi) = (IoU(Di, Pj) + DE(Di, Pj) + R(Di, Pj)) / 3,  for i ∈ D, j ∈ P
        """
        pass

    #Element-wise Weighted Mean of Every Cost Matrix Value (𝑊𝑀(𝐷,𝑃))
    def weighted_mean_cost_matrix(tracks, detections, image_dims, lambda_iou=0.33, lambda_de=0.33, lambda_r=0.34):
        """
        Computes the element-wise weighted mean of every cost matrix value:

        WM(Di, Pi) = (λ_IoU * IoU(Di, Pi) + λ_DE * DE(Di, Pi) + λ_R * R(Di, Pi)) / (λ_IoU + λ_DE + λ_R)

        where λ_IoU + λ_DE + λ_R = 1.
        """
        pass

    #Class Gate Update Based on Object Class Match (𝐶∗(𝐷,𝑃))
    def class_gate_cost_matrix(cost_matrix, track_classes, detection_classes):
        """
        Updates the cost matrix based on the match between predicted and detected
        object class. If the class labels do not match, the cost is set to infinity:

        C*(Ci, j, Di, Pi) = { Ci, j if Class_Di = Class_Pi, 0 otherwise }

        for i ∈ D, j ∈ P.
        """
        pass

