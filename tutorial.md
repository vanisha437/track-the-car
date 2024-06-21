Detect the Car in the Video 
============================

In this activity, you will learn to detect the car from an image and highlight them by drawing the boxes around it.

<img src= "https://media.slid.es/uploads/1525749/images/10501571/68pcp.gif" width = "180" height = "320">


Follow the given steps to complete this activity:
1. ### Create a tracker

* Open the main.py file.

*  Create `tracker` variable and set a tracker using `cv2.legacy.TrackerCSRT_create()`.
 
    `tracker = cv2.legacy.TrackerCSRT_create()`

* Initialize the tracker.

    `tracker.init(image, boxes[i])`
    
* Set `detected` variable to `True`.
 
     `detected = True`


* Get the `trackerInfo` from `tracker.update(image)`.
  
     `trackerInfo = tracker.update(image)`
     
 * Store `trackerInfor[0]` in success and `trackInfo[1]` in bbox variable.
  
            `success = trackerInfo[0]`
            
            `bbox = trackerInfo[1]`
            
* If success is true the call the `drawBox` function with the image and `bbox` variables.
 
            `if success:`
            
                `drawBox(image, bbox)`
                
* else add text "Lost" on the screen and set the detected variable to `false`.

          `else:`
          
                `cv2.putText(image, "Lost", (75, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)`
                            
               `detected = False`
               
* Save and run the code to check the output.


