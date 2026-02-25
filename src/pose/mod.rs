pub mod crop;
pub mod detector;
pub mod keypoint;
#[cfg(feature = "imgproc")]
pub mod person_detector;
pub mod preprocess;

pub use crop::{bbox_from_keypoints, remap_pose, BBox, CropRegion};
#[cfg(feature = "imgproc")]
pub use crop::crop_for_pose;
pub use detector::ModelType;
pub use detector::PoseDetector;
pub use keypoint::{Keypoint, KeypointIndex, Pose};
#[cfg(feature = "imgproc")]
pub use person_detector::PersonDetector;
#[cfg(feature = "imgproc")]
pub use preprocess::preprocess_for_movenet;
#[cfg(feature = "imgproc")]
pub use preprocess::preprocess_for_rtmw3d;
#[cfg(feature = "imgproc")]
pub use preprocess::preprocess_for_spinepose;
pub use preprocess::{unletterbox_pose, LetterboxInfo};
