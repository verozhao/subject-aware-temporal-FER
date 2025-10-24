# Robust Facial Emotion Recognition with Subject Conditioning and Continuous Facial Motion Detection
Robust Facial Emotion Recognition with Subject Conditioning and Continuous Facial Motion Detection
Facial emotion recognition (FER) is a critical technology for next-generation human-computer
interaction, enabling applications from adaptive tutoring systems to driver safety monitoring. For
these systems to be effective, their predictions must be not only accurate but also stable and reliable
over time. However, real-world conditions present significant challenges: faces are often partially
occluded, with constant variations in pose and lighting. Furthermore, existing models struggle to
generalize across different individuals and often fail to capture subtle, low-amplitude expressions.

While modern deep learning approaches have improved upon classical methods, they still have
key limitations. CNN-based models, though powerful, often produce unstable, frame-by-frame
predictions and perform poorly when faced with data from new subjects or environments. Sequence
models like RNNs can improve temporal consistency but at a high computational cost, making them
less suitable for real-time applications.

Two primary gaps remain in current research: (1) inadequate adaptation to the unique facial mor-
phology of each input, and (2) insufficient use of short-term facial dynamics, which are crucial for
distinguishing near-neutral expressions. Our project directly addresses these gaps. We propose to
build upon strong CNN baselines by introducing two key innovations: subject-specific conditioning
and a temporal model based on facial landmark motion, aiming to create a more robust and reliable
FER system for real-world applications.
