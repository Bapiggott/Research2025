# Drone Task Feasibility Summary

This document summarizes various drone task categories described in the reference PDF and indicates which tasks are planned, which tasks are uncertain, and which tasks are not applicable under the current setup.

---

## Overview

The tasks are grouped into categories such as **Pre-flight and Setup**, **Flight and Navigation**, and so on. For each category:

1. **Provided Notes** – Statements from the project notes on feasibility or interest in each type of task.  
2. **Feasibility** – A brief assessment of whether tasks are readily implementable or may require additional effort.  
3. **Additional Comments** – Any relevant clarifications or pointers based on the reference PDF.  
4. **Doc Reference** – The corresponding table/section in the PDF.

---

## Summary Table

| **Category**              | **Provided Notes**                                                                                                                                                                                                                           | **Feasibility**                                                               | **Additional Comments**                                                                                                 | **Doc Reference**      |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|------------------------|
| **Pre-flight & Setup**    | - Indicates capability to perform these tasks                                                                                                                                                                                                | **Yes** – Basic calibration, GPS lock, etc.                                   | Straightforward tasks (e.g., calibration, system checks). Likely matches Table 3, Section 3.1.                           | Table 3 (Sec. 3.1)     |
| **Flight & Navigation**   | - Capable of: takeoff, landing, hover, navigate to waypoint (x, y, z), return home, follow waypoint route<br/>- Uncertainty regarding path planning, obstacle avoidance, path re-planning                                                    | **Yes** for basic tasks<br/>**Maybe** for path/avoidance                      | Implementing direct flights (go_to_location, RTH, etc.) is simpler. Real-time path re-planning and avoidance require advanced sensor data.                    | Table 4 (Sec. 3.2)     |
| **Perception & Sensing**  | - Capable of: obstacle detection, object recognition, capturing images/video<br/>- Uncertainty regarding object tracking, SLAM (mapping), terrain following                                                                                 | **Yes** for basic detection<br/>**Maybe** for advanced (SLAM, tracking)       | Capturing images/video is straightforward. Object recognition is feasible if a trained model is available. SLAM/tracking require more extensive development.  | Table 5 (Sec. 3.3)     |
| **Communication**         | - Capable of: send telemetry, monitor battery, monitor link, geofence compliance<br/>- Not needed: receive external commands, stream camera data beyond IsaacSim<br/>- Uncertainty regarding emergency landing                              | **Yes** for telemetry & battery/link checks<br/>**No** for external streaming<br/>**Maybe** for emergency landing   | Sending status, battery levels, and geofence data is generally doable. Implementing emergency landing requires reliable fail-safe triggers and data sources.  | Table 6 (Sec. 3.4)     |
| **Safety & Health**       | - Indicates the need to monitor sensors (validity in IsaacSim)                                                                                                                                                                              | **Yes** – Basic sensor checks                                                 | Overlaps with Communication in tasks like battery and link monitoring. Safety can also include auto-land or other fail-safes if further development occurs.    | Table 6 (Sec. 3.4)     |
| **Advanced Landing**      | - Marked as N/A                                                                                                                                                                                                                            | **N/A**                                                                       | Precision or vision-based landing (beyond basic landing) is out of scope for the current environment.                                                          | Table 7 (Sec. 3.5)     |
| **Payload Handling**      | - Marked as N/A due to lack of a payload mechanism                                                                                                                                                                                          | **N/A**                                                                       | No hardware (servo/winch) to handle or release payloads.                                                                                                       | Table 7 (Sec. 3.5)     |
| **Swarm / Multi-UAV**     | - Marked as N/A since only one UAV is in operation                                                                                                                                                                                         | **N/A**                                                                       | Multi-drone coordination and collision avoidance among multiple vehicles is not applicable in a single-UAV scenario.                                           | Table 7 (Sec. 3.5)     |
| **Weather Adaptation**    | - Marked as N/A due to limitations of IsaacSim                                                                                                                                                                                             | **N/A**                                                                       | Weather-based path re-planning or adaptation is not relevant in IsaacSim at the present time.                                                                   | Table 7 (Sec. 3.5)     |

---

## Notes

1. **Basic Tasks** (e.g., pre-flight checks, GPS lock, takeoff, landing) are generally straightforward.  
2. **Navigation** can involve simple waypoint flights, hover, and return-to-home before attempting path planning or obstacle avoidance.  
3. **Advanced Capabilities** (path re-planning, obstacle avoidance, SLAM) typically require:
   - Additional sensors (e.g., LiDAR, depth camera).  
   - Real-time data processing pipelines.  
   - AI/ML models for advanced vision tasks.  
4. **Communication & Safety** (battery checks, link monitoring, geofence compliance) can be integrated early to manage fail-safes effectively.  
5. **Not Implementing** tasks such as advanced landing, payload handling, multi-UAV operations, and weather adaptation is acceptable given hardware/simulation constraints.

---

## Next Steps

- **Core Feature Implementation**: Begin with basic tasks (pre-flight, GPS lock, takeoff, waypoint navigation).  
- **Sensing Experiments**: Capture images/video and test object detection within IsaacSim.  
- **Further Evaluation**: Determine feasibility of path planning, obstacle avoidance, or SLAM if suitable sensor data and compute resources are available.

Refer to the PDF for additional details and example pseudocode in the Appendices (Sections A and B). 
