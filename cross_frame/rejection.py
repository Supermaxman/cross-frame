def check_results(response_schema, test_shuffled, test_problems):
    frames = response_schema["frames"]
    test_size = len(test_shuffled)

    seen_ids = set()
    for frame in frames:
        for post_id in frame["posts"]:
            if post_id in seen_ids:
                continue
            seen_ids.add(post_id)

    seen_problems = set()
    for frame in frames:
        for problem in frame["problems"]:
            if problem in seen_problems:
                continue
            seen_problems.add(problem)

    stats = {
        "total_posts": test_size,
        "total_problems": len(test_problems),
        "frames": len(frames),
        "posts": len(seen_ids),
        "problems": len(seen_problems),
    }

    if len(frames) == 0:
        print("No frames identified")
        return False, stats

    # Assume that there should be around 10% of the test_shuffled size frames
    # So for 750 posts there should be at least 75 frames
    # use 8% for leeway
    if len(frames) < test_size * 0.08:
        print(
            f"Only {len(frames)} frames identified for {test_size} posts ({len(frames)/test_size*100:.0f}%)"
        )
        return False, stats
    # A large proportion of the posts should be evoking some frame, say at least 50% for some leeway
    if len(seen_ids) < test_size * 0.50:
        print(
            f"Only {len(seen_ids)} posts identified for {test_size} posts ({len(seen_ids)/test_size*100:.0f}%)"
        )
        return False, stats

    # There should be good problem coverage - really all problems should be addressed, but for some leeway let's say
    # at least 50% of problems should be addressed by frames
    if len(seen_problems) < len(test_problems) * 0.50:
        print(
            f"Only {len(seen_problems)} problems addressed for {len(test_problems)} problems ({len(seen_problems)/len(test_problems)*100:.0f}%)"
        )
        return False, stats

    # Otherwise, should be valid sample
    return True, stats


def stats_to_text(stats):
    lines = []
    total_posts = stats["total_posts"]
    total_problems = stats["total_problems"]
    frames = stats["frames"]
    posts = stats["posts"]
    problems = stats["problems"]
    lines.append(
        f"Seen Posts: {posts}/{total_posts}({posts/total_posts*100:.0f}%), Seen Problems: {problems}/{total_problems}({problems/total_problems*100:.0f}%), Frames: {frames}"
    )
    return "\n".join(lines)
