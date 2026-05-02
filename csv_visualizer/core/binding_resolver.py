def resolve_binding(chart_type, data_shape):
    
    if chart_type in {"pie", "funnel", "pyramid"}:
        return "named_points"

    if chart_type in {"line", "area"} and data_shape == "time_series":
        return "datetime_series"

    if data_shape == "categorical_numeric":
        return "category_series"

    if data_shape == "two_dimensional_numeric":
        return "xy_pairs"

    if data_shape == "three_dimensional_numeric":
        return "xyz_points"

    if data_shape == "one_dimensional_numeric":
        return "series_y"

    if chart_type == "boxplot":
        return "boxplot_array"

    return "series_y"