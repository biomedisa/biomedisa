from django import template
register = template.Library()

@register.inclusion_tag("youtube_thumb.html")
def youtube_thumb(video_id, width=640):
    return {
        "video_id": video_id,
        "width": width,
    }

