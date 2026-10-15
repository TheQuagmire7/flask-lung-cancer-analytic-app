document.addEventListener("DOMContentLoaded", function () {

    // =====================================================
    // CHART
    // =====================================================

    const canvas = document.getElementById(
        "probabilityChart"
    );

    // Kalau belum prediction, chart memang belum ada.
    if (!canvas) {
        return;
    }


    // Pastikan data tersedia

    if (
        typeof chartLabels === "undefined" ||
        typeof chartProbabilities === "undefined"
    ) {
        return;
    }


    const ctx = canvas.getContext("2d");


    // =====================================================
    // CREATE CHART
    // =====================================================

    new Chart(ctx, {

        type: "bar",

        data: {

            labels: chartLabels,

            datasets: [

                {
                    label: "Risk Probability",

                    data: chartProbabilities,

                    borderWidth: 0,

                    borderRadius: 8,

                    barThickness: 45
                }

            ]

        },


        options: {

            responsive: true,

            maintainAspectRatio: false,


            plugins: {

                legend: {
                    display: false
                },

                tooltip: {

                    callbacks: {

                        label: function (context) {

                            return (
                                " Probability: " +
                                context.raw +
                                "%"
                            );

                        }

                    }

                }

            },


            scales: {

                y: {

                    beginAtZero: true,

                    max: 100,

                    ticks: {

                        stepSize: 20,

                        callback: function (value) {

                            return value + "%";

                        }

                    },

                    grid: {

                        color:
                            "rgba(148,163,184,0.15)"

                    }

                },


                x: {

                    grid: {

                        display: false

                    },

                    ticks: {

                        font: {
                            size: 11
                        }

                    }

                }

            }

        }

    });

});


// =========================================================
// RESET
// =========================================================

function resetDashboard() {

    const form =
        document.getElementById(
            "predictionForm"
        );

    if (form) {

        form.reset();

    }


    // Kembali ke halaman awal
    // supaya hasil prediction hilang.

    window.location.href = "/";

}