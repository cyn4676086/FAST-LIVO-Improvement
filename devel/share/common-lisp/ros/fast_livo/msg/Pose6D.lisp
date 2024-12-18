; Auto-generated. Do not edit!


(cl:in-package fast_livo-msg)


;//! \htmlinclude Pose6D.msg.html

(cl:defclass <Pose6D> (roslisp-msg-protocol:ros-message)
  ((offset_time
    :reader offset_time
    :initarg :offset_time
    :type cl:float
    :initform 0.0)
   (acc
    :reader acc
    :initarg :acc
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0))
   (gyr
    :reader gyr
    :initarg :gyr
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0))
   (vel
    :reader vel
    :initarg :vel
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0))
   (pos
    :reader pos
    :initarg :pos
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0))
   (rot
    :reader rot
    :initarg :rot
    :type (cl:vector cl:float)
   :initform (cl:make-array 9 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass Pose6D (<Pose6D>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Pose6D>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Pose6D)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name fast_livo-msg:<Pose6D> is deprecated: use fast_livo-msg:Pose6D instead.")))

(cl:ensure-generic-function 'offset_time-val :lambda-list '(m))
(cl:defmethod offset_time-val ((m <Pose6D>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader fast_livo-msg:offset_time-val is deprecated.  Use fast_livo-msg:offset_time instead.")
  (offset_time m))

(cl:ensure-generic-function 'acc-val :lambda-list '(m))
(cl:defmethod acc-val ((m <Pose6D>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader fast_livo-msg:acc-val is deprecated.  Use fast_livo-msg:acc instead.")
  (acc m))

(cl:ensure-generic-function 'gyr-val :lambda-list '(m))
(cl:defmethod gyr-val ((m <Pose6D>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader fast_livo-msg:gyr-val is deprecated.  Use fast_livo-msg:gyr instead.")
  (gyr m))

(cl:ensure-generic-function 'vel-val :lambda-list '(m))
(cl:defmethod vel-val ((m <Pose6D>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader fast_livo-msg:vel-val is deprecated.  Use fast_livo-msg:vel instead.")
  (vel m))

(cl:ensure-generic-function 'pos-val :lambda-list '(m))
(cl:defmethod pos-val ((m <Pose6D>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader fast_livo-msg:pos-val is deprecated.  Use fast_livo-msg:pos instead.")
  (pos m))

(cl:ensure-generic-function 'rot-val :lambda-list '(m))
(cl:defmethod rot-val ((m <Pose6D>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader fast_livo-msg:rot-val is deprecated.  Use fast_livo-msg:rot instead.")
  (rot m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Pose6D>) ostream)
  "Serializes a message object of type '<Pose6D>"
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'offset_time))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'acc))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'gyr))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'vel))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'pos))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'rot))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Pose6D>) istream)
  "Deserializes a message object of type '<Pose6D>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'offset_time) (roslisp-utils:decode-double-float-bits bits)))
  (cl:setf (cl:slot-value msg 'acc) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'acc)))
    (cl:dotimes (i 3)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-double-float-bits bits)))))
  (cl:setf (cl:slot-value msg 'gyr) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'gyr)))
    (cl:dotimes (i 3)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-double-float-bits bits)))))
  (cl:setf (cl:slot-value msg 'vel) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'vel)))
    (cl:dotimes (i 3)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-double-float-bits bits)))))
  (cl:setf (cl:slot-value msg 'pos) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'pos)))
    (cl:dotimes (i 3)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-double-float-bits bits)))))
  (cl:setf (cl:slot-value msg 'rot) (cl:make-array 9))
  (cl:let ((vals (cl:slot-value msg 'rot)))
    (cl:dotimes (i 9)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-double-float-bits bits)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Pose6D>)))
  "Returns string type for a message object of type '<Pose6D>"
  "fast_livo/Pose6D")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Pose6D)))
  "Returns string type for a message object of type 'Pose6D"
  "fast_livo/Pose6D")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Pose6D>)))
  "Returns md5sum for a message object of type '<Pose6D>"
  "ab486e9c24704038320abf9ff59003d2")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Pose6D)))
  "Returns md5sum for a message object of type 'Pose6D"
  "ab486e9c24704038320abf9ff59003d2")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Pose6D>)))
  "Returns full string definition for message of type '<Pose6D>"
  (cl:format cl:nil "# the preintegrated Lidar states at the time of IMU measurements in a frame~%float64  offset_time # the offset time of IMU measurement w.r.t the first lidar point~%float64[3] acc       # the preintegrated total acceleration (global frame) at the Lidar origin~%float64[3] gyr       # the unbiased angular velocity (body frame) at the Lidar origin~%float64[3] vel       # the preintegrated velocity (global frame) at the Lidar origin~%float64[3] pos       # the preintegrated position (global frame) at the Lidar origin~%float64[9] rot       # the preintegrated rotation (global frame) at the Lidar origin~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Pose6D)))
  "Returns full string definition for message of type 'Pose6D"
  (cl:format cl:nil "# the preintegrated Lidar states at the time of IMU measurements in a frame~%float64  offset_time # the offset time of IMU measurement w.r.t the first lidar point~%float64[3] acc       # the preintegrated total acceleration (global frame) at the Lidar origin~%float64[3] gyr       # the unbiased angular velocity (body frame) at the Lidar origin~%float64[3] vel       # the preintegrated velocity (global frame) at the Lidar origin~%float64[3] pos       # the preintegrated position (global frame) at the Lidar origin~%float64[9] rot       # the preintegrated rotation (global frame) at the Lidar origin~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Pose6D>))
  (cl:+ 0
     8
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'acc) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'gyr) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'vel) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'pos) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'rot) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Pose6D>))
  "Converts a ROS message object to a list"
  (cl:list 'Pose6D
    (cl:cons ':offset_time (offset_time msg))
    (cl:cons ':acc (acc msg))
    (cl:cons ':gyr (gyr msg))
    (cl:cons ':vel (vel msg))
    (cl:cons ':pos (pos msg))
    (cl:cons ':rot (rot msg))
))
