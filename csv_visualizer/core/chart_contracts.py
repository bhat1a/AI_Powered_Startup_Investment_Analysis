CHART_CONTRACTS = {

    # ─────────────────────────────────────────
    # Continuous Charts
    # ─────────────────────────────────────────

    "line": {
        "allowed_shapes": [
            "categorical_numeric",
            "two_dimensional_numeric",
            "time_series"
        ],
        "supports_grouping": True,
        "requires_sorted_x": False,
        "requires_module": None,
        "frontend_derived": False
    },

    "area": {
        "allowed_shapes": [
            "categorical_numeric",
            "two_dimensional_numeric",
            "time_series"
        ],
        "supports_grouping": True,
        "requires_sorted_x": False,
        "requires_module": None,
        "frontend_derived": False
    },

    "scatter": {
        "allowed_shapes": ["two_dimensional_numeric"],
        "supports_grouping": True,
        "requires_sorted_x": False,
        "requires_module": None,
        "frontend_derived": False
    },

    "bubble": {
        "allowed_shapes": ["three_dimensional_numeric"],
        "supports_grouping": True,
        "requires_sorted_x": False,
        "requires_module": "highcharts-more",
        "frontend_derived": False
    },


    # ─────────────────────────────────────────
    # Categorical Charts
    # ─────────────────────────────────────────

    "bar": {
        "allowed_shapes": [
            "categorical_numeric",
            "one_dimensional_numeric"
        ],
        "supports_grouping": True,
        "requires_sorted_x": False,
        "requires_module": None,
        "frontend_derived": False
    },

    "column": {
        "allowed_shapes": [
            "categorical_numeric",
            "one_dimensional_numeric"
        ],
        "supports_grouping": True,
        "requires_sorted_x": False,
        "requires_module": None,
        "frontend_derived": False
    },

    "pie": {
        "allowed_shapes": ["categorical_numeric"],
        "supports_grouping": False,
        "requires_sorted_x": False,
        "requires_module": None,
        "frontend_derived": False
    },


    # ─────────────────────────────────────────
    # Statistical Charts
    # ─────────────────────────────────────────

    "boxplot": {
        "allowed_shapes": ["categorical_numeric"],
        "supports_grouping": True,
        "requires_sorted_x": False,
        "requires_module": "highcharts-more",
        "frontend_derived": False
    },

    "histogram": {
        "allowed_shapes": ["one_dimensional_numeric"],
        "supports_grouping": False,
        "requires_sorted_x": False,
        "requires_module": "histogram-bellcurve",
        "frontend_derived": True
    },

    "bellcurve": {
        "allowed_shapes": ["one_dimensional_numeric"],
        "supports_grouping": False,
        "requires_sorted_x": False,
        "requires_module": "histogram-bellcurve",
        "frontend_derived": True
    },


    # ─────────────────────────────────────────
    # Hierarchical Charts
    # ─────────────────────────────────────────

    "treemap": {
        "allowed_shapes": ["hierarchical"],
        "supports_grouping": False,
        "requires_sorted_x": False,
        "requires_module": "treemap",
        "frontend_derived": False
    },

    "sunburst": {
        "allowed_shapes": ["hierarchical"],
        "supports_grouping": False,
        "requires_sorted_x": False,
        "requires_module": "sunburst",
        "frontend_derived": False
    },


    # ─────────────────────────────────────────
    # Matrix / 3D Mapping
    # ─────────────────────────────────────────

    "heatmap": {
        "allowed_shapes": ["three_dimensional_numeric"],
        "supports_grouping": False,
        "requires_sorted_x": False,
        "requires_module": "heatmap",
        "frontend_derived": False
    },


    # ─────────────────────────────────────────
    # Conversion Charts
    # ─────────────────────────────────────────

    "funnel": {
        "allowed_shapes": ["categorical_numeric"],
        "supports_grouping": False,
        "requires_sorted_x": False,
        "requires_module": "funnel",
        "frontend_derived": False
    },

    "pyramid": {
        "allowed_shapes": ["categorical_numeric"],
        "supports_grouping": False,
        "requires_sorted_x": False,
        "requires_module": "funnel",
        "frontend_derived": False
    }
}