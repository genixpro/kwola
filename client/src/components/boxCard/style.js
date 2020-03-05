import styled from 'styled-components';
import { palette } from 'styled-theme';
import { transition, boxShadow, borderRadius } from '../../settings/style-util';

const BoxCard = styled.div`
  padding: 20px 30px;
  margin-bottom: 10px;
  flex-shrink: 0;
  text-decoration: none;
  display: flex;
  flex-direction: volumn;
  text-decoration: none;
  cursor: pointer;
  background-color: #fff;
  text-align: ${props => (props['data-rtl'] === 'rtl' ? 'right' : 'left')};
  ${boxShadow('0 0px 3px 0px rgba(0,0,0,0.2)')};
  ${transition()};

  &:last-child {
    margin-bottom: 0;
  }

  .imgWrapper {
    width: 35px;
    height: 35px;
    overflow: hidden;
    margin: ${props =>
      props['data-rtl'] === 'rtl' ? '0 0 0 15px' : '0 15px 0 0'};
    display: -webkit-inline-flex;
    display: -ms-inline-flex;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    background-color: ${palette('grayscale', 9)};
    ${borderRadius('50%')};
    img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
  }

  .listContent {
    width: 100%;
    display: flex;
    flex-direction: column;

    .listHead {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }

    h5 {
      font-size: 14px;
      font-weight: 500;
      color: ${palette('grey', 8)};
      margin: 0;
      padding: ${props =>
        props['data-rtl'] === 'rtl' ? '0 0 0 15px' : '0 15px 0 0'};
    }

    p {
      font-size: 12px;
      font-weight: 400;
      color: ${palette('grey', 6)};
      line-height: 1.5;
      overflow: hidden;
      margin: 0;
      white-space: normal;
    }

    .date {
      font-size: 11px;
      color: ${palette('grey', 8)};
      flex-shrink: 0;
    }
  }
`;

export { BoxCard };
