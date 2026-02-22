#[cfg(feature = "desktop")]
pub mod crop;
pub mod detector;
pub mod keypoint;
#[cfg(feature = "desktop")]
pub mod person_detector;
#[cfg(feature = "desktop")]
pub mod preprocess;

#[cfg(feature = "desktop")]
pub use crop::{bbox_from_keypoints, crop_for_pose, remap_pose, BBox, CropRegion};
pub use detector::ModelType;
pub use detector::PoseDetector;
pub use keypoint::{Keypoint, KeypointIndex, Pose};
#[cfg(feature = "desktop")]
pub use person_detector::PersonDetector;
#[cfg(feature = "desktop")]
pub use preprocess::preprocess_for_movenet;
#[cfg(feature = "desktop")]
pub use preprocess::preprocess_for_rtmw3d;
#[cfg(feature = "desktop")]
pub use preprocess::preprocess_for_spinepose;
#[cfg(feature = "desktop")]
pub use preprocess::{unletterbox_pose, LetterboxInfo};
