import styled from 'styled-components';
import WithDirection from '../../settings/withDirection';

const ColorChooserDropdown = styled.div`
  display: flex;
  flex-flow: row wrap;

  .mateColorWrapper {
    justify-content: space-between;

    > span > div {
      min-width: 20px;
      min-height: 20px;
      position: relative;
      cursor: pointer;
      border: 0;
      margin: ${props =>
        props['data-rtl'] === 'rtl' ? '0 0 0 15px' : '0 15px 0 0'};
      padding: 0;
      border-radius: 3px;
    }
  }
`;

export default WithDirection(ColorChooserDropdown);
