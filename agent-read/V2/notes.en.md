# Notes

- Lift preserves grasp x/y and orientation and only increases z.
- Every segment starts from the preceding trajectory endpoint.
- `third_camera` is the right-side view; `opposite_top_camera` is the opposite overhead view.
- Legacy pickles are rejected and cannot be mixed with v2 replay.

## Foundation O.1.1 / O.1.2

- O.1 no longer resets object pose before close. A post-close state gate verifies that the object stayed stable and lies in the two-finger capture region before attaching a drive at the current pose.
- The default base-only `support_proxy` prevents the open gripper from tipping narrow bottle bodies during pregrasp/grasp.
- O.1.1 sets up from the first annotated keyframe. O.1.2 replaces lift/place with one action using second-keyframe EE xyz while retaining the grasp orientation.
