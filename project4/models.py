from django.db import models

class StudySession(models.Model):
    DESIGN_CHOICES = [
        ('pairwise', 'Pairwise comparisons'),
        ('ranking', 'Ranking of several movies'),
    ]

    design = models.CharField(max_length=20, choices=DESIGN_CHOICES)
    # Documented consent is a legal requirement for the study
    consent_given = models.BooleanField(default=False)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f'Session {self.pk} ({self.design})'


class Interaction(models.Model):
    study_session = models.ForeignKey(
        StudySession,
        on_delete=models.CASCADE,
        related_name='interactions',
    )
    position = models.PositiveIntegerField()

    # Row numbers into the feature matrix, in the order they were displayed
    shown_movie_indices = models.JSONField()
    # The same numbers in the order the participant preferred them with the best first
    # Stays empty until the participant answers, which also marks drop-outs
    answer_movie_indices = models.JSONField(null=True, blank=True)

    shown_at = models.DateTimeField(auto_now_add=True)
    answered_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['study_session', 'position']

    def __str__(self):
        return f'Interaction {self.position} of session {self.study_session_id}'
