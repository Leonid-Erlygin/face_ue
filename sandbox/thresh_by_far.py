# compute thresh by fars
probe_unique_ids = self.probe_pooled_templates[gallery_name][
    "template_subject_ids_sorted"
]
g_unique_ids = self.gallery_pooled_templates[gallery_name][
    "template_subject_ids_sorted"
]
is_seen = np.isin(probe_unique_ids, g_unique_ids)
probe_score = np.max(similarity[:, 0, :], axis=1)
neg_score = probe_score[~is_seen]
neg_score_sorted = np.sort(neg_score)[::-1]
for far in [0.005, 0.01, 0.05, 0.1]:
    thresh = neg_score_sorted[max(int((neg_score_sorted.shape[0]) * far) - 1, 0)]
    print(f"FAR-{far}_thresh:{thresh}")
