const theme = {};

theme.palette = {
  primary: [],
  secondary: [],
  shadows: [
    'none', // 0
    '0px 1px 3px 0px rgba(0, 0, 0, 0.2),0px 1px 1px 0px rgba(0, 0, 0, 0.14),0px 2px 1px -1px rgba(0, 0, 0, 0.12)', // 1 -> Elevation 1
    '0px 1px 5px 0px rgba(0, 0, 0, 0.2),0px 2px 2px 0px rgba(0, 0, 0, 0.14),0px 3px 1px -2px rgba(0, 0, 0, 0.12)', // 2 -> Elevation 2
    '0px 1px 8px 0px rgba(0, 0, 0, 0.2),0px 3px 4px 0px rgba(0, 0, 0, 0.14),0px 3px 3px -2px rgba(0, 0, 0, 0.12)', // 3 -> Elevation 3
    '0px 2px 4px -1px rgba(0, 0, 0, 0.2),0px 4px 5px 0px rgba(0, 0, 0, 0.14),0px 1px 10px 0px rgba(0, 0, 0, 0.12)', // 4 -> Elevation 4
    '0px 3px 5px -1px rgba(0, 0, 0, 0.2),0px 5px 8px 0px rgba(0, 0, 0, 0.14),0px 1px 14px 0px rgba(0, 0, 0, 0.12)', // 5 -> Elevation 5
    '0px 3px 5px -1px rgba(0, 0, 0, 0.2),0px 6px 10px 0px rgba(0, 0, 0, 0.14),0px 1px 18px 0px rgba(0, 0, 0, 0.12)', // 6 -> Elevation 6
    '0px 4px 5px -2px rgba(0, 0, 0, 0.2),0px 7px 10px 1px rgba(0, 0, 0, 0.14),0px 2px 16px 1px rgba(0, 0, 0, 0.12)', // 7 -> Elevation 7
    '0px 5px 5px -3px rgba(0, 0, 0, 0.2),0px 8px 10px 1px rgba(0, 0, 0, 0.14),0px 3px 14px 2px rgba(0, 0, 0, 0.12)', // 8 -> Elevation 8
    '0px 5px 6px -3px rgba(0, 0, 0, 0.2),0px 9px 12px 1px rgba(0, 0, 0, 0.14),0px 3px 16px 2px rgba(0, 0, 0, 0.12)', // 9 -> Elevation 9
    '0px 6px 6px -3px rgba(0, 0, 0, 0.2),0px 10px 14px 1px rgba(0, 0, 0, 0.14),0px 4px 18px 3px rgba(0, 0, 0, 0.12)', // 10 -> Elevation 10
    '0px 6px 7px -4px rgba(0, 0, 0, 0.2),0px 11px 15px 1px rgba(0, 0, 0, 0.14),0px 4px 20px 3px rgba(0, 0, 0, 0.12)', // 11 -> Elevation 11
    '0px 7px 8px -4px rgba(0, 0, 0, 0.2),0px 12px 17px 2px rgba(0, 0, 0, 0.14),0px 5px 22px 4px rgba(0, 0, 0, 0.12)', // 12 -> Elevation 12
    '0px 7px 8px -4px rgba(0, 0, 0, 0.2),0px 13px 19px 2px rgba(0, 0, 0, 0.14),0px 5px 24px 4px rgba(0, 0, 0, 0.12)', // 13 -> Elevation 13
    '0px 7px 9px -4px rgba(0, 0, 0, 0.2),0px 14px 21px 2px rgba(0, 0, 0, 0.14),0px 5px 26px 4px rgba(0, 0, 0, 0.12)', // 14 -> Elevation 14
    '0px 8px 9px -5px rgba(0, 0, 0, 0.2),0px 15px 22px 2px rgba(0, 0, 0, 0.14),0px 6px 28px 5px rgba(0, 0, 0, 0.12)', // 15 -> Elevation 15
    '0px 8px 10px -5px rgba(0, 0, 0, 0.2),0px 16px 24px 2px rgba(0, 0, 0, 0.14),0px 6px 30px 5px rgba(0, 0, 0, 0.12)', // 16 -> Elevation 16
    '0px 8px 11px -5px rgba(0, 0, 0, 0.2),0px 17px 26px 2px rgba(0, 0, 0, 0.14),0px 6px 32px 5px rgba(0, 0, 0, 0.12)', // 17 -> Elevation 17
    '0px 9px 11px -5px rgba(0, 0, 0, 0.2),0px 18px 28px 2px rgba(0, 0, 0, 0.14),0px 7px 34px 6px rgba(0, 0, 0, 0.12)', // 18 -> Elevation 18
    '0px 9px 12px -6px rgba(0, 0, 0, 0.2),0px 19px 29px 2px rgba(0, 0, 0, 0.14),0px 7px 36px 6px rgba(0, 0, 0, 0.12)', // 19 -> Elevation 19
    '0px 10px 13px -6px rgba(0, 0, 0, 0.2),0px 20px 31px 3px rgba(0, 0, 0, 0.14),0px 8px 38px 7px rgba(0, 0, 0, 0.12)', // 20 -> Elevation 20
    '0px 10px 13px -6px rgba(0, 0, 0, 0.2),0px 21px 33px 3px rgba(0, 0, 0, 0.14),0px 8px 40px 7px rgba(0, 0, 0, 0.12)', // 21 -> Elevation 21
    '0px 10px 14px -6px rgba(0, 0, 0, 0.2),0px 22px 35px 3px rgba(0, 0, 0, 0.14),0px 8px 42px 7px rgba(0, 0, 0, 0.12)', // 22 -> Elevation 22
    '0px 11px 14px -7px rgba(0, 0, 0, 0.2),0px 23px 36px 3px rgba(0, 0, 0, 0.14),0px 9px 44px 8px rgba(0, 0, 0, 0.12)', // 23 -> Elevation 23
    '0px 11px 15px -7px rgba(0, 0, 0, 0.2),0px 24px 38px 3px rgba(0, 0, 0, 0.14),0px 9px 46px 8px rgba(0, 0, 0, 0.12)', // 24 -> Elevation 24
  ],
  blue: [
    '#E3F2FD', // 0 - 50
    '#BBDEFB', // 1 - 100
    '#90CAF9', // 2 - 200
    '#64B5F6', // 3 - 300
    '#42A5F5', // 4 - 400
    '#2196F3', // 5 - 500
    '#1E88E5', // 6 - 600
    '#1976D2', // 7 - 700
    '#1565C0', // 8 - 800
    '#0D47A1', // 9 - 900
    '#82B1FF', // 10 - A100
    '#448AFF', // 11 - A200
    '#2979FF', // 12 - A400
    '#2962FF', // 13 - A700
    '#278BE1', // 14 - custom topbar & sidebar selected
    '#1d89e4', // 15
    '#3369e7', // 16 - Algolia
  ],
  red: [
    '#FFEBEE', // 0 - 50
    '#FFCDD2', // 1 - 100
    '#EF9A9A', // 2 - 200
    '#E57373', // 3 - 300
    '#EF5350', // 4 - 400
    '#F44336', // 5 - 500
    '#E53935', // 6 - 600
    '#D32F2F', // 7 - 700
    '#C62828', // 8 - 800
    '#B71C1C', // 9 - 900
    '#FF8A80', // 10 - A100
    '#FF5252', // 11 - A200
    '#FF1744', // 12 - A400
    '#D50000', // 13 - A700
  ],
  orange: [
    '#FFF3E0', // 0 - 50
    '#FFE0B2', // 1 - 100
    '#FFCC80', // 2 - 200
    '#FFB74D', // 3 - 300
    '#FFA726', // 4 - 400
    '#FF9800', // 5 - 500
    '#FB8C00', // 6 - 600
    '#F57C00', // 7 - 700
    '#EF6C00', // 8 - 800
    '#E65100', // 9 - 900
    '#FFD180', // 10 - A100
    '#FFAB40', // 11 - A200
    '#FF9100', // 12 - A400
    '#FF6D00', // 13 - A700
    '#feac02', // 14 - Rating Color
  ],
  purple: [
    '#F3E5F5', // 0 - 50
    '#E1BEE7', // 1 - 100
    '#CE93D8', // 2 - 200
    '#BA68C8', // 3 - 300
    '#AB47BC', // 4 - 400
    '#9C27B0', // 5 - 500
    '#8E24AA', // 6 - 600
    '#7B1FA2', // 7 - 700
    '#6A1B9A', // 8 - 800
    '#4A148C', // 9 - 900
    '#EA80FC', // 10 - A100
    '#E040FB', // 11 - A200
    '#D500F9', // 12 - A400
    '#AA00FF', // 13 - A700
  ],
  green: [
    '#E8F5E9', // 0 - 50
    '#C8E6C9', // 1 - 100
    '#A5D6A7', // 2 - 200
    '#81C784', // 3 - 300
    '#66BB6A', // 4 - 400
    '#4CAF50', // 5 - 500
    '#43A047', // 6 - 600
    '#388E3C', // 7 - 700
    '#2E7D32', // 8 - 800
    '#1B5E20', // 9 - 900
    '#B9F6CA', // 10 - A100
    '#69F0AE', // 11 - A200
    '#00E676', // 12 - A400
    '#00C853', // 13 - A700
  ],
  pink: [
    '#FCE4EC', // 0 - 50
    '#F8BBD0', // 1 - 100
    '#F48FB1', // 2 - 200
    '#F06292', // 3 - 300
    '#EC407A', // 4 - 400
    '#E91E63', // 5 - 500
    '#D81B60', // 6 - 600
    '#C2185B', // 7 - 700
    '#AD1457', // 8 - 800
    '#880E4F', // 9 - 900
    '#FF80AB', // 10 - A100
    '#FF4081', // 11 - A200
    '#F50057', // 12 - A400
    '#C51162', // 13 - A700
  ],
  indigo: [
    '#E8EAF6', // 0 - 50
    '#C5CAE9', // 1 - 100
    '#9FA8DA', // 2 - 200
    '#7986CB', // 3 - 300
    '#5C6BC0', // 4 - 400
    '#3F51B5', // 5 - 500 text color
    '#3949AB', // 6 - 600
    '#303F9F', // 7 - 700
    '#283593', // 8 - 800 heading color
    '#1A237E', // 9 - 900
    '#8C9EFF', // 10 - A100
    '#536DFE', // 11 - A200
    '#3D5AFE', // 12 - A400
    '#304FFE', // 13 - A700
  ],
  grey: [
    '#FAFAFA', // 0 - 50
    '#F5F5F5', // 1 - 100
    '#EEEEEE', // 2 - 200
    '#E0E0E0', // 3 - 300
    '#BDBDBD', // 4 - 400
    '#9E9E9E', // 5 - 500 text color
    '#757575', // 6 - 600
    '#616161', // 7 - 700
    '#424242', // 8 - 800 heading color
    '#212121', // 9 - 900
    '#2B2B2B', // 10 - custom
    '#262626', // 11 - Expand menu
    '#ECECEC', // 12 - custom bg
  ],

  warning: [],
  success: [],
  error: [],
  text: [
    '#9E9E9E', // 0 - Sidebar text Color
  ],
  calendar: [
    '#30C26C', // 0
    '#FF4C6C', // 1
    '#FFE34D', // 2
    '#EDEEEE', // 3
  ],
  pages: [
    '#424242', // 0
    '#616161', // 1
    '#303f9f', // 2
    '#283593', // 3
    '#757575', // 4
    '#2979FF', // 5
    '#49a9ee', // 6
    '#3b5998', // 7 (facebook signup)
    '#dd4b39', // 8 (googleplus signup)
    '#e14615', // 9 (authzero signup)
    '#FFCA28', // 10 (firebase signup)
    '#385490', // 11 (facebook signup hover)
    '#da402d', // 12 (googleplus signup hover)
    '#d54213', // 13 (authzero signup hover)
    '#fec619', // 14 (firebase signup hover)
  ],
  border: [
    '#e9e9e9', //0 // notes border color
  ],
};

theme.fonts = {
  primary: 'Roboto, sans-serif',
  pre: 'Consolas, Liberation Mono, Menlo, Courier, monospace',
};

export default theme;
